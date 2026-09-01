"""Regression tests for transactional gradient centralization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_centralization import centralize_gradients_
from engine.tensor import Tensor


def _parameter(shape, gradient):
    parameter = Tensor(np.zeros(shape, dtype=np.float64), requires_grad=True)
    parameter.grad = np.asarray(gradient).copy()
    return parameter


def test_rank2_gradient_is_centered_per_leading_unit():
    parameter = _parameter((2, 3), [[1.0, 2.0, 6.0], [-4.0, 2.0, 8.0]])
    gradient = parameter.grad

    changed = centralize_gradients_([parameter])

    assert changed == 1
    assert parameter.grad is gradient
    np.testing.assert_allclose(
        parameter.grad,
        [[-2.0, -1.0, 3.0], [-6.0, 0.0, 6.0]],
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(parameter.grad.mean(axis=1), 0.0, atol=1e-15, rtol=0.0)


def test_rank3_centers_all_nonleading_axes_as_one_unit():
    parameter = _parameter((2, 2, 2), np.arange(8.0).reshape(2, 2, 2))

    changed = centralize_gradients_(parameter)

    assert changed == 1
    flattened = parameter.grad.reshape(2, -1)
    np.testing.assert_allclose(flattened.mean(axis=1), 0.0, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(flattened[0], [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(flattened[1], [-1.5, -0.5, 0.5, 1.5])


def test_vectors_missing_gradients_and_empty_collection_are_noops():
    vector = _parameter((3,), [1.0, 2.0, 3.0])
    before = vector.grad.copy()
    missing = Tensor(np.zeros((2, 2)), requires_grad=True)
    missing.grad = None

    assert centralize_gradients_([vector, missing]) == 0
    assert centralize_gradients_([]) == 0
    np.testing.assert_array_equal(vector.grad, before)


def test_min_rank_can_limit_centralization_to_higher_rank_gradients():
    matrix = _parameter((2, 2), [[1.0, 3.0], [4.0, 8.0]])
    before = matrix.grad.copy()

    assert centralize_gradients_([matrix], min_rank=3) == 0
    np.testing.assert_array_equal(matrix.grad, before)


@pytest.mark.parametrize(
    ("shape", "gradient", "min_rank"),
    [
        ((2,), [1.0, 2.0], 2),
        ((1, 2), [[1.0, 3.0]], 3),
    ],
)
def test_ineligible_frozen_parameter_with_live_gradient_is_rejected(
    shape, gradient, min_rank
):
    parameter = Tensor(np.zeros(shape, dtype=np.float64), requires_grad=False)
    parameter.grad = np.asarray(gradient, dtype=np.float64).copy()
    before = parameter.grad.copy()

    with pytest.raises(ValueError, match="parameter 0 is frozen but still has a gradient"):
        centralize_gradients_([parameter], min_rank=min_rank)

    np.testing.assert_array_equal(parameter.grad, before)


@pytest.mark.parametrize(
    ("gradient", "error", "message"),
    [
        (np.array([1.0]), ValueError, "shape mismatch"),
        (np.array([1, 2], dtype=np.int64), TypeError, "floating dtype"),
        (np.array([1.0, np.nan]), ValueError, "finite values"),
    ],
)
def test_ineligible_live_gradient_is_still_validated(gradient, error, message):
    parameter = Tensor(np.zeros((2,), dtype=np.float64), requires_grad=True)
    parameter.grad = gradient
    binding = parameter.grad
    before = np.array(binding, copy=True)

    with pytest.raises(error, match=message):
        centralize_gradients_([parameter])

    assert parameter.grad is binding
    np.testing.assert_array_equal(parameter.grad, before)


def test_zero_mean_gradient_is_not_rewritten():
    parameter = _parameter((2, 2), [[-1.0, 1.0], [-3.0, 3.0]])
    gradient = parameter.grad

    assert centralize_gradients_([parameter]) == 0
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [[-1.0, 1.0], [-3.0, 3.0]])


def test_huge_same_sign_values_compute_mean_without_sum_overflow():
    largest = np.finfo(np.float64).max
    parameter = _parameter((1, 2), [[largest, largest]])

    with np.errstate(all="raise"):
        changed = centralize_gradients_([parameter])

    assert changed == 1
    np.testing.assert_array_equal(parameter.grad, np.zeros((1, 2)))


def test_unrepresentable_centered_value_fails_before_any_write():
    largest = np.finfo(np.float64).max
    first = _parameter((1, 2), [[1.0, 3.0]])
    second = _parameter((1, 3), [[largest, largest, -largest]])
    first_before = first.grad.copy()
    second_before = second.grad.copy()

    with pytest.raises(ValueError, match="centralized gradient"):
        centralize_gradients_([first, second])

    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)


def test_nonfinite_late_gradient_fails_before_any_write():
    first = _parameter((1, 2), [[1.0, 3.0]])
    second = _parameter((1, 2), [[1.0, np.nan]])
    first_before = first.grad.copy()

    with pytest.raises(ValueError, match="gradient for parameter 1.*finite"):
        centralize_gradients_([first, second])

    np.testing.assert_array_equal(first.grad, first_before)


def test_read_only_gradient_is_allowed_for_noop_but_rejected_for_write():
    noop = _parameter((1, 2), [[-1.0, 1.0]])
    noop.grad.setflags(write=False)
    assert centralize_gradients_([noop]) == 0

    active = _parameter((1, 2), [[1.0, 3.0]])
    active.grad.setflags(write=False)
    before = active.grad.copy()
    with pytest.raises(ValueError, match="parameter 0.*writable"):
        centralize_gradients_([active])
    np.testing.assert_array_equal(active.grad, before)


def test_overlapping_changed_gradients_are_rejected_before_write():
    storage = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    first = Tensor(np.zeros((1, 2)), requires_grad=True)
    second = Tensor(np.zeros((1, 2)), requires_grad=True)
    first.grad = storage[:2].reshape(1, 2)
    second.grad = storage[1:].reshape(1, 2)
    before = storage.copy()

    with pytest.raises(ValueError, match="must not overlap"):
        centralize_gradients_([first, second])

    np.testing.assert_array_equal(storage, before)


def test_changed_gradient_must_not_alias_its_parameter_data():
    parameter = Tensor(np.array([[1.0, 3.0]], dtype=np.float64), requires_grad=True)
    parameter.grad = parameter.data
    data = parameter.data
    before = np.array(data, copy=True)
    version = parameter._version

    with pytest.raises(ValueError, match="gradient.*parameter 0 data"):
        centralize_gradients_([parameter])

    assert parameter.data is data
    assert parameter.grad is data
    assert parameter._version == version
    np.testing.assert_array_equal(parameter.data, before)


def test_changed_gradient_must_not_alias_another_bound_parameter_data():
    first = Tensor(np.array([[1.0, 3.0]], dtype=np.float64), requires_grad=True)
    first.grad = np.array([[2.0, 6.0]], dtype=np.float64)
    second = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    second.grad = first.data
    first_grad_before = first.grad.copy()
    first_data_before = np.array(first.data, copy=True)
    first_version = first._version

    with pytest.raises(ValueError, match="gradient for parameter 1.*parameter 0 data"):
        centralize_gradients_([first, second])

    np.testing.assert_array_equal(first.grad, first_grad_before)
    np.testing.assert_array_equal(first.data, first_data_before)
    assert first._version == first_version


def test_noop_gradient_parameter_data_alias_is_allowed_without_write():
    parameter = Tensor(np.array([[-1.0, 1.0]], dtype=np.float64), requires_grad=True)
    parameter.grad = parameter.data
    data = parameter.data
    version = parameter._version

    assert centralize_gradients_([parameter]) == 0

    assert parameter.data is data
    assert parameter.grad is data
    assert parameter._version == version
    np.testing.assert_array_equal(parameter.data, [[-1.0, 1.0]])


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1, 0, -2, 2.5, "2"])
def test_min_rank_validation_is_explicit(bad):
    parameter = _parameter((1, 2), [[1.0, 3.0]])
    before = parameter.grad.copy()

    with pytest.raises((TypeError, ValueError), match="min_rank"):
        centralize_gradients_([parameter], min_rank=bad)

    np.testing.assert_array_equal(parameter.grad, before)


def test_duplicate_and_non_tensor_parameters_are_rejected():
    parameter = _parameter((1, 2), [[1.0, 3.0]])

    with pytest.raises(ValueError, match="duplicate"):
        centralize_gradients_([parameter, parameter])
    with pytest.raises(TypeError, match="parameter 0"):
        centralize_gradients_([np.zeros((1, 2))])
