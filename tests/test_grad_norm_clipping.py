"""Correctness tests for stable global gradient-norm clipping."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import clip_grad_norm_
from engine.tensor import Tensor


def _parameter(values, gradient):
    parameter = Tensor(values, requires_grad=True)
    parameter.grad[:] = gradient
    return parameter


def test_clip_grad_norm_scales_all_active_gradients_by_one_global_factor():
    first = _parameter([1.0], [3.0])
    second = _parameter([2.0], [4.0])
    first_grad = first.grad
    second_grad = second.grad

    total_norm = clip_grad_norm_([first, second], 2.0)

    assert total_norm == 5.0
    assert first.grad is first_grad
    assert second.grad is second_grad
    np.testing.assert_allclose(first.grad, [1.2])
    np.testing.assert_allclose(second.grad, [1.6])
    assert np.hypot(first.grad[0], second.grad[0]) == pytest.approx(2.0)


def test_norm_below_limit_is_bitwise_unchanged():
    parameter = _parameter([1.0, 2.0], [0.25, -0.5])
    gradient = parameter.grad
    before = gradient.copy()

    total_norm = clip_grad_norm_([parameter], 10.0)

    assert total_norm == pytest.approx(np.sqrt(0.3125))
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, before)


def test_none_gradients_are_ignored_and_zero_norm_is_a_noop():
    inactive = Tensor([1.0], requires_grad=True)
    inactive.grad = None
    zero = _parameter([2.0, 3.0], [0.0, 0.0])
    before = zero.grad.copy()

    assert clip_grad_norm_([inactive, zero], 1.0) == 0.0
    assert inactive.grad is None
    np.testing.assert_array_equal(zero.grad, before)


def test_zero_max_norm_zeros_nonzero_active_gradients():
    parameter = _parameter([1.0, 2.0], [3.0, -4.0])

    total_norm = clip_grad_norm_([parameter], 0.0)

    assert total_norm == 5.0
    np.testing.assert_array_equal(parameter.grad, [0.0, 0.0])


def test_unrepresentable_global_norm_still_gets_meaningful_clipping_scale():
    first = _parameter([1.0], [1.4e308])
    second = _parameter([1.0], [1.4e308])

    with np.errstate(all="raise"):
        total_norm = clip_grad_norm_([first, second], 1.0)

    assert np.isinf(total_norm)
    expected = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(first.grad, [expected], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(second.grad, [expected], rtol=1e-15, atol=0.0)
    assert np.hypot(first.grad[0], second.grad[0]) == pytest.approx(1.0)


def test_parameter_iterable_is_materialized_once():
    parameters = [
        _parameter([1.0], [3.0]),
        _parameter([2.0], [4.0]),
    ]

    total_norm = clip_grad_norm_((parameter for parameter in parameters), 5.0)

    assert total_norm == 5.0
    np.testing.assert_array_equal(parameters[0].grad, [3.0])
    np.testing.assert_array_equal(parameters[1].grad, [4.0])


def test_duplicate_parameter_rejection_is_transactional():
    parameter = _parameter([1.0], [3.0])
    before = parameter.grad.copy()

    with pytest.raises(ValueError, match="duplicate at index 1"):
        clip_grad_norm_([parameter, parameter], 1.0)

    np.testing.assert_array_equal(parameter.grad, before)


@pytest.mark.parametrize(
    ("max_norm", "error", "message"),
    [
        (True, TypeError, "real number"),
        (-1.0, ValueError, "at least 0.0"),
        (np.nan, ValueError, "finite"),
        (np.inf, ValueError, "finite"),
    ],
)
def test_invalid_max_norm_is_rejected_before_gradient_mutation(max_norm, error, message):
    parameter = _parameter([1.0], [3.0])
    before = parameter.grad.copy()

    with pytest.raises(error, match=message):
        clip_grad_norm_([parameter], max_norm)

    np.testing.assert_array_equal(parameter.grad, before)


def test_late_invalid_gradient_leaves_earlier_gradients_unchanged():
    first = _parameter([1.0, 2.0], [3.0, 4.0])
    second = _parameter([1.0, 2.0], [1.0, 2.0])
    second.grad[1] = np.nan
    before_first = first.grad.copy()
    before_second = second.grad.copy()

    with pytest.raises(ValueError, match="parameter 1.*finite values"):
        clip_grad_norm_([first, second], 1.0)

    np.testing.assert_array_equal(first.grad, before_first)
    np.testing.assert_array_equal(second.grad, before_second)


def test_integer_gradient_rejection_is_transactional():
    first = _parameter([1.0], [3.0])
    second = Tensor([2.0], requires_grad=True)
    second.grad = np.array([4], dtype=np.int64)
    before_first = first.grad.copy()
    before_second = second.grad.copy()

    with pytest.raises(TypeError, match="parameter 1.*floating dtype"):
        clip_grad_norm_([first, second], 1.0)

    np.testing.assert_array_equal(first.grad, before_first)
    np.testing.assert_array_equal(second.grad, before_second)
