import numpy as np
import pytest

from engine.gradient_accumulator import GradientAccumulator
from engine.tensor import Tensor


def test_weighted_microbatch_average_and_copy_to_grads():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    accumulator = GradientAccumulator([first, second])

    first.grad[...] = [1.0, 3.0]
    second.grad = None
    first_grad = first.grad
    accumulator.accumulate(weight=2)

    # Accumulation is detached from the live gradient slots.
    np.testing.assert_array_equal(first.grad, [1.0, 3.0])
    assert first.grad is first_grad
    assert second.grad is None

    first.grad[...] = [5.0, 7.0]
    second.grad = np.array([4.0], dtype=np.float64)
    accumulator.accumulate(weight=1)

    average_first, average_second = accumulator.average_gradients()
    np.testing.assert_allclose(average_first, [7.0 / 3.0, 13.0 / 3.0])
    np.testing.assert_allclose(average_second, [4.0 / 3.0])
    assert accumulator.total_weight == 3.0
    assert accumulator.accumulation_count == 2

    accumulator.copy_to_grads()
    np.testing.assert_allclose(first.grad, average_first)
    np.testing.assert_allclose(second.grad, average_second)
    assert first.grad.dtype == np.float64
    assert second.grad.dtype == np.float64
    assert first.grad is not average_first
    assert second.grad is not average_second


def test_missing_gradient_is_zero_contribution_not_previous_value():
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)

    parameter.grad[...] = [8.0]
    accumulator.accumulate()
    parameter.grad = None
    accumulator.accumulate()

    (average,) = accumulator.average_gradients()
    np.testing.assert_array_equal(average, [4.0])


def test_accumulate_is_transactional_when_late_gradient_is_bad():
    first = Tensor([0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator([first, second])

    first.grad[...] = [2.0]
    second.grad[...] = [4.0]
    accumulator.accumulate()
    before = accumulator.state_dict()

    first.grad[...] = [100.0]
    second.grad[...] = [np.nan]
    with pytest.raises(ValueError, match="gradient 1 must contain only finite values"):
        accumulator.accumulate(weight=3.0)

    after = accumulator.state_dict()
    assert after["total_weight"] == before["total_weight"]
    assert after["accumulation_count"] == before["accumulation_count"]
    for actual, expected in zip(after["averages"], before["averages"]):
        np.testing.assert_array_equal(actual, expected)


def test_extreme_finite_online_average_avoids_sum_overflow():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0], requires_grad=True)
    accumulator = GradientAccumulator(parameter)

    with np.errstate(all="raise"):
        parameter.grad[...] = [maximum]
        accumulator.accumulate()
        parameter.grad[...] = [maximum]
        accumulator.accumulate()
        (same_sign,) = accumulator.average_gradients()
    np.testing.assert_array_equal(same_sign, [maximum])

    accumulator.reset()
    with np.errstate(all="raise"):
        parameter.grad[...] = [1.3e308]
        accumulator.accumulate()
        parameter.grad[...] = [-1.3e308]
        accumulator.accumulate()
        (opposite_sign,) = accumulator.average_gradients()
    np.testing.assert_allclose(opposite_sign, [0.0], atol=0.0)


def test_float32_gradients_are_normalized_to_independent_float64_buffers():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.array([1.5, -2.5], dtype=np.float32)
    accumulator = GradientAccumulator(parameter)

    accumulator.accumulate()
    parameter.grad[...] = 99.0
    (average,) = accumulator.average_gradients()

    assert average.dtype == np.float64
    np.testing.assert_array_equal(average, [1.5, -2.5])


def test_reset_does_not_touch_live_gradients():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [3.0]
    accumulator = GradientAccumulator(parameter)
    accumulator.accumulate(weight=2.0)
    live = parameter.grad

    accumulator.reset()

    assert accumulator.total_weight == 0.0
    assert accumulator.accumulation_count == 0
    assert parameter.grad is live
    np.testing.assert_array_equal(parameter.grad, [3.0])
    with pytest.raises(RuntimeError, match="no gradients have been accumulated"):
        accumulator.average_gradients()
    with pytest.raises(RuntimeError, match="no gradients have been accumulated"):
        accumulator.copy_to_grads()


def test_parameter_data_versions_and_rng_are_neutral():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [5.0]
    accumulator = GradientAccumulator(parameter)
    version = parameter._version
    data = parameter.data.copy()

    np.random.seed(1234)
    expected = np.random.get_state()
    accumulator.accumulate()
    accumulator.average_gradients()
    accumulator.copy_to_grads()
    actual = np.random.get_state()

    assert parameter._version == version
    np.testing.assert_array_equal(parameter.data, data)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_empty_parameter_collection_is_supported():
    accumulator = GradientAccumulator([])
    assert accumulator.accumulate(weight=2.0) == 2.0
    assert accumulator.average_gradients() == ()
    accumulator.copy_to_grads()
    accumulator.reset()
    assert accumulator.accumulation_count == 0
