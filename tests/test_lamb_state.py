import copy
import sys

import numpy as np
import pytest

from engine.lamb import LAMB
from engine.tensor import Tensor


def _make_optimizer(parameter):
    return LAMB(
        parameter,
        lr=0.03,
        betas=(0.6, 0.8),
        eps=1e-5,
        weight_decay=0.02,
    )


def test_state_round_trip_preserves_hyperparameters_and_moments():
    parameter = Tensor([2.0, -3.0], requires_grad=True)
    optimizer = _make_optimizer(parameter)
    parameter.grad = np.array([1.5, -0.5])
    optimizer.step()

    state = optimizer.state_dict()
    restored_parameter = Tensor(parameter.data.copy(), requires_grad=True)
    restored = LAMB(restored_parameter)
    restored.load_state_dict(state)
    restored_state = restored.state_dict()

    assert restored_state["version"] == state["version"] == 1
    assert restored_state["type"] == state["type"] == "LAMB"
    assert restored_state["lr"] == state["lr"]
    assert restored_state["betas"] == state["betas"]
    assert restored_state["eps"] == state["eps"]
    assert restored_state["weight_decay"] == state["weight_decay"]
    assert restored_state["states"][0]["step"] == 1
    np.testing.assert_array_equal(
        restored_state["states"][0]["m"], state["states"][0]["m"]
    )
    np.testing.assert_array_equal(
        restored_state["states"][0]["v"], state["states"][0]["v"]
    )
    assert restored_state["states"][0]["v_scale"] == state["states"][0]["v_scale"]


def test_resumed_optimizer_matches_next_step_trajectory():
    original_parameter = Tensor([4.0, -2.0, 1.0], requires_grad=True)
    original = _make_optimizer(original_parameter)
    for gradient in (
        np.array([1.0, -3.0, 2.0]),
        np.array([-2.0, 1.0, 4.0]),
    ):
        original_parameter.grad = gradient
        original.step()

    checkpoint = original.state_dict()
    resumed_parameter = Tensor(original_parameter.data.copy(), requires_grad=True)
    resumed = LAMB(resumed_parameter)
    resumed.load_state_dict(checkpoint)

    next_gradient = np.array([3.0, -5.0, 0.25])
    original_parameter.grad = next_gradient.copy()
    resumed_parameter.grad = next_gradient.copy()
    original.step()
    resumed.step()

    np.testing.assert_array_equal(resumed_parameter.data, original_parameter.data)
    original_state = original.state_dict()["states"][0]
    resumed_state = resumed.state_dict()["states"][0]
    assert resumed_state["step"] == original_state["step"]
    assert resumed_state["v_scale"] == original_state["v_scale"]
    np.testing.assert_array_equal(resumed_state["m"], original_state["m"])
    np.testing.assert_array_equal(resumed_state["v"], original_state["v"])


def test_state_dict_arrays_are_independent_copies():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    parameter.grad = np.array([1.0, 2.0])
    optimizer = _make_optimizer(parameter)
    optimizer.step()

    exported = optimizer.state_dict()
    internal_before = optimizer.state_dict()
    exported["states"][0]["m"][...] = 999.0
    exported["states"][0]["v"][...] = 999.0

    after = optimizer.state_dict()
    np.testing.assert_array_equal(after["states"][0]["m"], internal_before["states"][0]["m"])
    np.testing.assert_array_equal(after["states"][0]["v"], internal_before["states"][0]["v"])


def test_float32_state_arrays_normalize_to_float64():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = LAMB(parameter)
    state = optimizer.state_dict()
    state["states"][0] = {
        "step": 1,
        "m": np.array([1.0, -2.0], dtype=np.float32),
        "v_scale": 2.0,
        "v": np.array([0.25, 1.0], dtype=np.float32),
    }

    optimizer.load_state_dict(state)
    loaded = optimizer.state_dict()["states"][0]

    assert loaded["m"].dtype == np.float64
    assert loaded["v"].dtype == np.float64
    np.testing.assert_allclose(loaded["m"], [1.0, -2.0])
    np.testing.assert_allclose((loaded["v_scale"] ** 2) * loaded["v"], [1.0, 4.0])


def test_noncanonical_second_moment_is_canonicalized_without_changing_physical_values():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = LAMB(parameter)
    state = optimizer.state_dict()
    state["states"][0] = {
        "step": 3,
        "m": np.array([0.5, -0.25]),
        "v_scale": 2.0,
        "v": np.array([4.0, 1.0]),
    }

    optimizer.load_state_dict(state)
    loaded = optimizer.state_dict()["states"][0]

    np.testing.assert_allclose((loaded["v_scale"] ** 2) * loaded["v"], [16.0, 4.0])
    assert np.max(loaded["v"]) == pytest.approx(1.0)
    assert loaded["v_scale"] == pytest.approx(4.0)


def test_extended_precision_state_outside_float64_is_rejected_transactionally():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range")
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = LAMB(parameter)
    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["states"][0] = {
        "step": 1,
        "m": np.array([np.finfo(np.longdouble).max], dtype=np.longdouble),
        "v_scale": 1.0,
        "v": np.array([1.0]),
    }

    with pytest.raises(ValueError, match="fit float64"):
        optimizer.load_state_dict(malformed)

    after = optimizer.state_dict()
    assert after["lr"] == before["lr"]
    assert after["states"][0]["step"] == 0
    np.testing.assert_array_equal(after["states"][0]["m"], before["states"][0]["m"])


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda s: s.update(version=2), ValueError),
        (lambda s: s.update(type="Adam"), ValueError),
        (lambda s: s.update(lr=0.0), ValueError),
        (lambda s: s.update(betas=(0.9, 1.0)), ValueError),
        (lambda s: s.update(eps=False), TypeError),
        (lambda s: s.update(weight_decay=-1.0), ValueError),
        (lambda s: s.update(states=[]), ValueError),
    ],
)
def test_malformed_state_envelope_is_transactional(mutation, error):
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = LAMB(parameter)
    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    mutation(malformed)

    with pytest.raises(error):
        optimizer.load_state_dict(malformed)

    after = optimizer.state_dict()
    assert after["lr"] == before["lr"]
    assert after["betas"] == before["betas"]
    assert after["states"][0]["step"] == 0


def test_step_zero_requires_empty_moments():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = LAMB(parameter)
    state = optimizer.state_dict()
    state["states"][0]["m"] = np.array([1.0])

    with pytest.raises(ValueError, match="unused LAMB state"):
        optimizer.load_state_dict(state)


def test_zero_second_scale_requires_zero_normalized_buffer():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = LAMB(parameter)
    state = optimizer.state_dict()
    state["states"][0] = {
        "step": 1,
        "m": np.array([0.0]),
        "v_scale": 0.0,
        "v": np.array([1.0]),
    }

    with pytest.raises(ValueError, match="zero second-moment scale"):
        optimizer.load_state_dict(state)


def test_unknown_extra_metadata_is_tolerated():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = LAMB(parameter)
    state = optimizer.state_dict()
    state["future_metadata"] = {"ignored": True}
    state["states"][0]["future_field"] = "ignored"

    optimizer.load_state_dict(state)

    assert optimizer.steps == (0,)


def test_maximum_supported_step_loads_but_next_active_step_fails_before_data_write():
    parameter = Tensor([2.0], requires_grad=True)
    optimizer = LAMB(parameter, lr=0.1)
    state = optimizer.state_dict()
    state["states"][0] = {
        "step": sys.maxsize,
        "m": np.array([0.0]),
        "v_scale": 0.0,
        "v": np.array([0.0]),
    }
    optimizer.load_state_dict(state)
    parameter.grad = np.array([1.0])
    before = parameter.data.copy()

    with pytest.raises(OverflowError, match="step maximum"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, before)
    assert optimizer.steps == (sys.maxsize,)
