import copy

import numpy as np
import pytest

from engine.adafactor import Adafactor
from engine.tensor import Tensor


_TINY = np.nextafter(0.0, 1.0)


def _make_optimizer(parameter):
    return Adafactor(
        parameter,
        lr=0.03,
        beta2=0.75,
        eps=1e-20,
        clip_threshold=0.8,
    )


def test_state_round_trip_resumes_identical_vector_trajectory():
    original_parameter = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    original = _make_optimizer(original_parameter)

    original_parameter.grad = np.array([2.0, -1.0, 4.0])
    original.step()
    saved_parameter = original_parameter.data.copy()
    saved_state = original.state_dict()

    restored_parameter = Tensor(saved_parameter, requires_grad=True)
    restored = Adafactor(restored_parameter)
    restored.load_state_dict(saved_state)

    next_gradient = np.array([-3.0, 5.0, 1.0])
    original_parameter.grad = next_gradient.copy()
    restored_parameter.grad = next_gradient.copy()
    original.step()
    restored.step()

    np.testing.assert_array_equal(restored_parameter.data, original_parameter.data)
    restored_state = restored.state_dict()
    original_state = original.state_dict()
    assert restored_state["lr"] == original_state["lr"]
    assert restored_state["beta2"] == original_state["beta2"]
    assert restored_state["eps"] == original_state["eps"]
    assert restored_state["clip_threshold"] == original_state["clip_threshold"]
    assert restored.steps == original.steps == (2,)
    np.testing.assert_array_equal(
        restored_state["states"][0]["v"], original_state["states"][0]["v"]
    )
    assert restored_state["states"][0]["scale"] == original_state["states"][0]["scale"]


def test_state_round_trip_resumes_identical_factored_trajectory():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    original_parameter = Tensor(values, requires_grad=True)
    original = _make_optimizer(original_parameter)
    original_parameter.grad = np.arange(2.0, 14.0).reshape(3, 4)
    original.step()

    restored_parameter = Tensor(original_parameter.data.copy(), requires_grad=True)
    restored = Adafactor(restored_parameter)
    restored.load_state_dict(original.state_dict())

    gradient = np.arange(-6.0, 6.0).reshape(3, 4)
    original_parameter.grad = gradient.copy()
    restored_parameter.grad = gradient.copy()
    original.step()
    restored.step()

    np.testing.assert_array_equal(restored_parameter.data, original_parameter.data)
    a = original.state_dict()["states"][0]
    b = restored.state_dict()["states"][0]
    assert a["kind"] == b["kind"] == "factored"
    assert a["step"] == b["step"] == 2
    assert a["scale"] == b["scale"]
    np.testing.assert_array_equal(a["row"], b["row"])
    np.testing.assert_array_equal(a["col"], b["col"])


def test_state_dict_returns_independent_arrays():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([3.0, 4.0])
    optimizer = _make_optimizer(p)
    optimizer.step()

    exported = optimizer.state_dict()
    exported["states"][0]["v"][...] = 999.0

    fresh = optimizer.state_dict()
    assert not np.all(fresh["states"][0]["v"] == 999.0)


def test_load_state_does_not_write_model_or_gradient():
    p = Tensor([7.0, 8.0], requires_grad=True)
    gradient = np.array([2.0, 3.0])
    p.grad = gradient
    optimizer = _make_optimizer(p)

    source_p = Tensor([1.0, 2.0], requires_grad=True)
    source_p.grad = np.array([4.0, 5.0])
    source = _make_optimizer(source_p)
    source.step()
    state = source.state_dict()

    before_data = p.data.copy()
    before_version = p._version
    optimizer.load_state_dict(state)

    np.testing.assert_array_equal(p.data, before_data)
    assert p._version == before_version
    assert p.grad is gradient
    np.testing.assert_array_equal(gradient, [2.0, 3.0])


def test_load_float32_buffers_normalizes_to_float64():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([3.0, 4.0])
    source = _make_optimizer(p)
    source.step()
    state = source.state_dict()
    state["states"][0]["v"] = state["states"][0]["v"].astype(np.float32)

    restored = Adafactor(Tensor([1.0, 2.0], requires_grad=True))
    restored.load_state_dict(state)

    assert restored.state_dict()["states"][0]["v"].dtype == np.float64


def test_load_noncanonical_positive_buffers_are_canonicalized_without_changing_physical_moment():
    p = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adafactor(p)
    state = optimizer.state_dict()
    state["states"][0] = {
        "kind": "full",
        "step": 3,
        "scale": 2.0,
        "v": np.array([4.0, 1.0]),
    }

    optimizer.load_state_dict(state)
    loaded = optimizer.state_dict()["states"][0]

    assert loaded["scale"] == pytest.approx(4.0)
    np.testing.assert_allclose(loaded["v"], [1.0, 0.25])
    np.testing.assert_allclose(
        loaded["v"] * loaded["scale"] ** 2,
        [16.0, 4.0],
    )


def test_load_rejects_wrong_envelope_and_state_count_transactionally():
    p = Tensor([1.0], requires_grad=True)
    p.grad = np.array([1.0])
    optimizer = _make_optimizer(p)
    optimizer.step()
    before = optimizer.state_dict()

    cases = []
    bad = copy.deepcopy(before)
    bad["version"] = 999
    cases.append((bad, "version"))
    bad = copy.deepcopy(before)
    bad["type"] = "Other"
    cases.append((bad, "type"))
    bad = copy.deepcopy(before)
    bad["states"] = []
    cases.append((bad, "count"))

    for bad_state, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            optimizer.load_state_dict(bad_state)
        after = optimizer.state_dict()
        assert after["lr"] == before["lr"]
        assert after["states"][0]["step"] == before["states"][0]["step"]
        np.testing.assert_array_equal(
            after["states"][0]["v"], before["states"][0]["v"]
        )


def test_load_rejects_invalid_hyperparameters_without_state_change():
    p = Tensor([1.0], requires_grad=True)
    optimizer = Adafactor(p)
    before = optimizer.state_dict()

    for field, value in [
        ("lr", 0.0),
        ("beta2", 1.0),
        ("eps", 0.0),
        ("clip_threshold", np.inf),
    ]:
        bad = copy.deepcopy(before)
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            optimizer.load_state_dict(bad)
        assert optimizer.state_dict()[field] == before[field]


def test_load_rejects_kind_shape_dtype_finiteness_and_negative_moments():
    p = Tensor([1.0, 2.0], requires_grad=True)
    p.grad = np.array([1.0, 2.0])
    source = _make_optimizer(p)
    source.step()
    baseline = source.state_dict()

    target = Adafactor(Tensor([1.0, 2.0], requires_grad=True))

    bad = copy.deepcopy(baseline)
    bad["states"][0]["kind"] = "factored"
    with pytest.raises(ValueError, match="kind"):
        target.load_state_dict(bad)

    bad = copy.deepcopy(baseline)
    bad["states"][0]["v"] = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="shape"):
        target.load_state_dict(bad)

    bad = copy.deepcopy(baseline)
    bad["states"][0]["v"] = np.array([1, 2], dtype=np.int64)
    target.load_state_dict(bad)
    assert target.state_dict()["states"][0]["v"].dtype == np.float64

    bad = copy.deepcopy(baseline)
    bad["states"][0]["v"] = np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        target.load_state_dict(bad)

    bad = copy.deepcopy(baseline)
    bad["states"][0]["v"] = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        target.load_state_dict(bad)


def test_load_rejects_impossible_unused_and_active_state_invariants():
    p = Tensor([1.0], requires_grad=True)
    optimizer = Adafactor(p)
    base = optimizer.state_dict()

    bad = copy.deepcopy(base)
    bad["states"][0]["scale"] = 1.0
    with pytest.raises(ValueError, match="unused"):
        optimizer.load_state_dict(bad)

    bad = copy.deepcopy(base)
    bad["states"][0]["step"] = 1
    with pytest.raises(ValueError, match="active"):
        optimizer.load_state_dict(bad)

    bad = copy.deepcopy(base)
    bad["states"][0]["step"] = 1
    bad["states"][0]["scale"] = 1.0
    with pytest.raises(ValueError, match="active"):
        optimizer.load_state_dict(bad)


def test_load_rejects_extended_precision_moment_outside_float64_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([1.0], requires_grad=True)
    optimizer = Adafactor(p)
    state = optimizer.state_dict()
    huge = np.array([np.finfo(np.float64).max], dtype=np.longdouble)
    huge *= np.longdouble(2)
    state["states"][0] = {
        "kind": "full",
        "step": 1,
        "scale": 1.0,
        "v": huge,
    }

    with pytest.raises(ValueError, match="fit float64"):
        optimizer.load_state_dict(state)


def test_unknown_extra_metadata_is_tolerated():
    p = Tensor([1.0], requires_grad=True)
    optimizer = Adafactor(p)
    state = optimizer.state_dict()
    state["future_metadata"] = {"note": "ignored"}
    state["states"][0]["future"] = 123

    assert optimizer.load_state_dict(state) is optimizer
