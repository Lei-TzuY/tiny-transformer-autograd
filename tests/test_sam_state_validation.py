import copy

import numpy as np
import pytest

from engine.optim import Adam, SGD
from engine.sam import SAM
from engine.tensor import Tensor


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
        return
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return
    assert left == right


def test_extended_precision_gradient_that_cannot_fit_float64_is_rejected_cleanly():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64")

    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([np.longdouble(np.finfo(np.float64).max) * 2], dtype=np.longdouble)
    optimizer = SAM(SGD([parameter]), rho=0.1)
    version = parameter._version

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="must fit in float64"):
            optimizer.first_step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert parameter._version == version
    assert optimizer.phase == "ready"


def test_float32_gradient_is_snapshotted_and_supported():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([3.0, 4.0], dtype=np.float32)
    optimizer = SAM(SGD([parameter]), rho=0.5)

    optimizer.first_step()

    np.testing.assert_allclose(parameter.data, [1.3, 2.4], rtol=0.0, atol=1e-15)
    optimizer.restore()


def test_subnormal_gradient_is_warning_neutral_under_strict_numpy_errors():
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [tiny]
    optimizer = SAM(SGD([parameter]), rho=tiny)

    with np.errstate(all="raise"):
        optimizer.first_step()
        optimizer.restore()

    np.testing.assert_array_equal(parameter.data, [0.0])


@pytest.mark.parametrize(
    "mutator, error, message",
    [
        (lambda state: state.update(version=True), TypeError, "state version"),
        (lambda state: state.update(version=2), ValueError, "unsupported SAM state version"),
        (lambda state: state.update(optimizer_type=3), TypeError, "optimizer_type"),
        (lambda state: state.update(optimizer_type="AdamW"), ValueError, "optimizer type mismatch"),
        (lambda state: state.update(rho=True), TypeError, "SAM rho"),
        (lambda state: state.update(rho=-0.1), ValueError, "non-negative"),
        (lambda state: state.update(rho=np.inf), ValueError, "finite"),
        (lambda state: state.update(step_count=True), TypeError, "step_count"),
        (lambda state: state.update(step_count=-1), ValueError, "step_count"),
    ],
)
def test_rejected_metadata_loads_leave_sam_and_inner_state_unchanged(
    mutator, error, message
):
    parameter = Tensor([1.0], requires_grad=True)
    inner = Adam([parameter], lr=0.01)
    optimizer = SAM(inner, rho=0.2)
    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    mutator(malformed)

    with pytest.raises(error, match=message):
        optimizer.load_state_dict(malformed)

    _assert_nested_equal(optimizer.state_dict(), before)


def test_missing_optimizer_state_is_rejected_before_metadata_commit():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(SGD([parameter]), rho=0.2)
    before = optimizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["rho"] = 0.8
    del malformed["optimizer"]

    with pytest.raises(KeyError, match="optimizer"):
        optimizer.load_state_dict(malformed)

    _assert_nested_equal(optimizer.state_dict(), before)


def test_non_mapping_state_is_rejected_explicitly():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(SGD([parameter]))

    for state in (None, [], (), "state"):
        with pytest.raises(TypeError, match="state must be a dictionary"):
            optimizer.load_state_dict(state)


def test_state_dict_does_not_checkpoint_fast_model_parameters():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SAM(SGD([parameter]), rho=0.1)
    saved = optimizer.state_dict()

    parameter.data[...] = [9.0]
    optimizer.load_state_dict(saved)

    np.testing.assert_array_equal(parameter.data, [9.0])


def test_state_dict_inner_arrays_are_independent_in_both_directions():
    parameter = Tensor([1.0], requires_grad=True)
    inner = Adam([parameter])
    optimizer = SAM(inner)

    state = optimizer.state_dict()
    state["optimizer"]["m"][0][...] = [7.0]
    np.testing.assert_array_equal(inner._m[0], [0.0])

    inner._m[0][...] = [3.0]
    np.testing.assert_array_equal(state["optimizer"]["m"][0], [7.0])
