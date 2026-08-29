import copy
import json

import numpy as np
import pytest

from engine.gradient_accumulator import GradientAccumulator
from engine.tensor import Tensor


def _state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key == "averages":
            for actual, expected in zip(left[key], right[key]):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert left[key] == right[key]


def test_state_round_trip_reproduces_future_weighted_average():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    original = GradientAccumulator(parameter)
    parameter.grad[...] = [2.0, 6.0]
    original.accumulate(weight=2.0)
    state = original.state_dict()

    restored = GradientAccumulator(parameter)
    restored.load_state_dict(state)

    parameter.grad[...] = [8.0, -2.0]
    original.accumulate(weight=3.0)
    restored.accumulate(weight=3.0)

    np.testing.assert_array_equal(
        original.average_gradients()[0], restored.average_gradients()[0]
    )
    assert original.total_weight == restored.total_weight
    assert original.accumulation_count == restored.accumulation_count


def test_state_dict_owns_independent_average_copies():
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    parameter.grad[...] = [3.0]
    accumulator.accumulate()

    first = accumulator.state_dict()
    second = accumulator.state_dict()
    first["averages"][0][...] = 99.0

    np.testing.assert_array_equal(second["averages"][0], [3.0])
    np.testing.assert_array_equal(accumulator.average_gradients()[0], [3.0])


def test_rejected_load_is_transactional_and_does_not_touch_live_gradients():
    first = Tensor([0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator([first, second])
    first.grad[...] = [1.0]
    second.grad[...] = [2.0]
    accumulator.accumulate()
    before = accumulator.state_dict()
    first_live = first.grad
    second_live = second.grad

    bad = accumulator.state_dict()
    bad["averages"][1][...] = np.nan
    with pytest.raises(ValueError, match="average 1 must contain only finite values"):
        accumulator.load_state_dict(bad)

    _state_equal(accumulator.state_dict(), before)
    assert first.grad is first_live
    assert second.grad is second_live
    np.testing.assert_array_equal(first.grad, [1.0])
    np.testing.assert_array_equal(second.grad, [2.0])


@pytest.mark.parametrize(
    "mutate, exc_type, match",
    [
        (lambda state: state.update(version=True), ValueError, "unsupported"),
        (lambda state: state.update(version=2), ValueError, "unsupported"),
        (lambda state: state.update(type=1), ValueError, "type mismatch"),
        (lambda state: state.update(type="Other"), ValueError, "type mismatch"),
        (lambda state: state.update(total_weight=-1.0), ValueError, "non-negative"),
        (lambda state: state.update(total_weight=np.inf), ValueError, "finite"),
        (lambda state: state.update(accumulation_count=True), TypeError, "integer"),
        (lambda state: state.update(accumulation_count=-1), ValueError, "non-negative"),
    ],
)
def test_state_metadata_validation(mutate, exc_type, match):
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    state = accumulator.state_dict()
    mutate(state)
    with pytest.raises(exc_type, match=match):
        accumulator.load_state_dict(state)


def test_state_count_weight_invariants_are_validated():
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)

    state = accumulator.state_dict()
    state["accumulation_count"] = 1
    with pytest.raises(ValueError, match="count/weight state is inconsistent"):
        accumulator.load_state_dict(state)

    state = accumulator.state_dict()
    state["total_weight"] = 1.0
    with pytest.raises(ValueError, match="count/weight state is inconsistent"):
        accumulator.load_state_dict(state)

    state = accumulator.state_dict()
    state["averages"][0][...] = 1.0
    with pytest.raises(ValueError, match="empty gradient accumulator state must contain zero averages"):
        accumulator.load_state_dict(state)


def test_average_envelope_shape_dtype_and_count_validation():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    state = accumulator.state_dict()

    bad = copy.deepcopy(state)
    bad["averages"] = "bad"
    with pytest.raises(TypeError, match="averages must be a list or tuple"):
        accumulator.load_state_dict(bad)

    bad = copy.deepcopy(state)
    bad["averages"] = []
    with pytest.raises(ValueError, match="average count mismatch"):
        accumulator.load_state_dict(bad)

    bad = copy.deepcopy(state)
    bad["averages"][0] = [0.0, 0.0]
    with pytest.raises(TypeError, match="must be a NumPy array"):
        accumulator.load_state_dict(bad)

    bad = copy.deepcopy(state)
    bad["averages"][0] = np.zeros((1,), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        accumulator.load_state_dict(bad)

    bad = copy.deepcopy(state)
    bad["averages"][0] = np.zeros((2,), dtype=np.int64)
    with pytest.raises(TypeError, match="floating-point"):
        accumulator.load_state_dict(bad)


def test_float32_state_is_normalized_and_unknown_keys_are_tolerated():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    state = {
        "version": 1,
        "type": "GradientAccumulator",
        "total_weight": np.float32(2.0),
        "accumulation_count": np.int64(1),
        "averages": [np.array([1.5, -2.5], dtype=np.float32)],
        "future_metadata": {"ignored": True},
    }
    accumulator.load_state_dict(state)
    average = accumulator.average_gradients()[0]
    assert average.dtype == np.float64
    np.testing.assert_array_equal(average, [1.5, -2.5])


def test_state_is_not_json_safe_by_design_but_metadata_is_plain_python():
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    parameter.grad[...] = [1.0]
    accumulator.accumulate()
    state = accumulator.state_dict()

    assert type(state["version"]) is int
    assert type(state["type"]) is str
    assert type(state["total_weight"]) is float
    assert type(state["accumulation_count"]) is int
    with pytest.raises(TypeError):
        json.dumps(state)


def test_extended_precision_state_that_cannot_fit_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)
    huge = np.array([np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)], dtype=np.longdouble)
    assert np.isfinite(huge).all()
    state = {
        "version": 1,
        "type": "GradientAccumulator",
        "total_weight": 1.0,
        "accumulation_count": 1,
        "averages": [huge],
    }
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="must fit in float64"):
            accumulator.load_state_dict(state)
