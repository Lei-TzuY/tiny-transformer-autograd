import copy

import numpy as np
import pytest

from engine.lookahead import Lookahead
from engine.optim import Adam, SGD
from engine.tensor import Tensor


def _state_snapshot(optimizer):
    state = optimizer.state_dict()
    return copy.deepcopy(state)


def _assert_state_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
        return
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        np.testing.assert_array_equal(left, right)
        return
    assert left == right


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("version", 2, ValueError),
        ("version", True, TypeError),
        ("optimizer_type", "SGD", ValueError),
        ("optimizer_type", 3, TypeError),
        ("sync_period", 0, ValueError),
        ("sync_period", True, TypeError),
        ("alpha", np.inf, ValueError),
        ("alpha", True, TypeError),
        ("step_count", -1, ValueError),
        ("step_count", 1.5, TypeError),
        ("pending_sync", 1, TypeError),
    ],
)
def test_invalid_wrapper_metadata_is_rejected_before_inner_state_mutation(
    field, value, exception
):
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lookahead(Adam([parameter]), sync_period=3, alpha=0.4)
    before = _state_snapshot(optimizer)
    malformed = copy.deepcopy(before)
    malformed[field] = value
    malformed["optimizer"]["lr"] = 0.25

    with pytest.raises(exception):
        optimizer.load_state_dict(malformed)

    _assert_state_equal(optimizer.state_dict(), before)


def test_slow_weight_count_and_shape_validation_precede_inner_state_mutation():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0, 3.0], requires_grad=True)
    optimizer = Lookahead(SGD([first, second]), sync_period=2)
    before = _state_snapshot(optimizer)

    wrong_count = copy.deepcopy(before)
    wrong_count["slow_weights"] = wrong_count["slow_weights"][:1]
    wrong_count["optimizer"]["lr"] = 0.2
    with pytest.raises(ValueError, match="slow weight count mismatch"):
        optimizer.load_state_dict(wrong_count)
    _assert_state_equal(optimizer.state_dict(), before)

    wrong_shape = copy.deepcopy(before)
    wrong_shape["slow_weights"][1] = np.zeros((3,), dtype=np.float64)
    wrong_shape["optimizer"]["lr"] = 0.2
    with pytest.raises(ValueError, match="shape mismatch"):
        optimizer.load_state_dict(wrong_shape)
    _assert_state_equal(optimizer.state_dict(), before)


def test_extended_precision_slow_weight_overflow_is_rejected_without_commit():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range than float64")

    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lookahead(SGD([parameter]), sync_period=2)
    before = _state_snapshot(optimizer)
    malformed = copy.deepcopy(before)
    malformed["slow_weights"][0] = np.array(
        [np.longdouble(np.finfo(np.float64).max) * np.longdouble(2.0)],
        dtype=np.longdouble,
    )

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="must fit in float64"):
            optimizer.load_state_dict(malformed)

    _assert_state_equal(optimizer.state_dict(), before)


def test_load_state_dict_requires_dictionary():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = Lookahead(SGD([parameter]))

    with pytest.raises(TypeError, match="state must be a dictionary"):
        optimizer.load_state_dict([])
