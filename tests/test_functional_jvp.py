"""Functional Jacobian-vector products should be correct and state-isolated."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine import jacobian, jvp
from engine.tensor import Tensor


def test_jvp_scalar_output_matches_directional_derivative():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)

    result = jvp(output, x, np.array([5.0, 7.0]))

    assert result.shape == ()
    np.testing.assert_allclose(result, np.array(62.0))


def test_jvp_vector_output_preserves_output_shape():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x + 3.0 * x
    tangent = np.array([5.0, 7.0])

    result = jvp(output, x, tangent)

    np.testing.assert_allclose(result, (2.0 * np.asarray(x.data) + 3.0) * tangent)


def test_jvp_single_element_iterable_inputs_require_matching_tangent_iterable():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x
    tangent = np.array([5.0, 7.0])

    result = jvp(output, (x,), (tangent,))

    np.testing.assert_allclose(result, 2.0 * np.asarray(x.data) * tangent)


def test_jvp_multiple_inputs_adds_joint_direction_contributions():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([5.0, 7.0], requires_grad=True)
    output = x * y + x
    tangent_x = np.array([11.0, 13.0])
    tangent_y = np.array([17.0, 19.0])

    result = jvp(output, (x, y), (tangent_x, tangent_y))

    expected = tangent_x * (np.asarray(y.data) + 1.0) + tangent_y * np.asarray(x.data)
    np.testing.assert_allclose(result, expected)


def test_jvp_matches_dense_jacobian_contraction():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 7.0], [11.0, 13.0]], requires_grad=True)
    output = (x * y + x).transpose((1, 0))
    tangent_x = np.array([[0.5, -1.0], [1.5, 2.0]])
    tangent_y = np.array([[2.0, 3.0], [-0.5, 4.0]])

    jac_x, jac_y = jacobian(output, (x, y))
    result = jvp(output, (x, y), (tangent_x, tangent_y))

    output_axes = tuple(range(output.ndim, output.ndim + x.ndim))
    input_axes = tuple(range(x.ndim))
    expected = np.tensordot(jac_x, tangent_x, axes=(output_axes, input_axes))
    expected += np.tensordot(jac_y, tangent_y, axes=(output_axes, input_axes))
    np.testing.assert_allclose(result, expected)


def test_jvp_preserves_existing_gradient_buffers_and_values():
    x = Tensor([2.0, 3.0], requires_grad=True)
    hidden = x * x
    output = hidden + x
    x.grad[:] = np.array([17.0, 19.0])
    hidden.grad[:] = np.array([23.0, 29.0])
    output.grad[:] = np.array([31.0, 37.0])

    x_buffer = x.grad
    hidden_buffer = hidden.grad
    output_buffer = output.grad
    x_before = x.grad.copy()
    hidden_before = hidden.grad.copy()
    output_before = output.grad.copy()

    result = jvp(output, x, np.array([5.0, 7.0]))

    np.testing.assert_allclose(result, np.array([25.0, 49.0]))
    assert x.grad is x_buffer
    assert hidden.grad is hidden_buffer
    assert output.grad is output_buffer
    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(hidden.grad, hidden_before)
    np.testing.assert_array_equal(output.grad, output_before)


def test_jvp_duplicate_inputs_accumulate_both_requested_directions():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x
    first = np.array([5.0, 7.0])
    second = np.array([11.0, 13.0])

    result = jvp(output, (x, x), (first, second))

    expected = 2.0 * np.asarray(x.data) * (first + second)
    np.testing.assert_allclose(result, expected)


def test_jvp_empty_output_still_returns_correct_shape():
    x = Tensor(np.empty((0, 2)), requires_grad=True)
    output = x * 3.0

    result = jvp(output, x, np.empty((0, 2)))

    assert result.shape == (0, 2)
    assert result.dtype == np.float64


def test_jvp_recovers_finite_direction_after_product_overflow_cancellation():
    x = Tensor([0.0, 0.0], requires_grad=True)
    coefficients = Tensor([1e308, 1e308])
    output = ops.sum(x * coefficients)

    with np.errstate(all="raise"):
        result = jvp(output, x, np.array([3.0, -3.0]))

    assert result == 0.0


def test_jvp_preserves_genuinely_unrepresentable_directional_derivative():
    x = Tensor([0.0, 0.0], requires_grad=True)
    coefficients = Tensor([1e308, 1e308])
    output = ops.sum(x * coefficients)

    with np.errstate(all="raise"):
        result = jvp(output, x, np.array([3.0, 3.0]))

    assert np.isposinf(result)


def test_jvp_rejects_stale_graph_without_mutating_gradients():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)
    x.grad[:] = np.array([5.0, 7.0])
    output.grad[...] = 11.0
    x_buffer = x.grad
    output_buffer = output.grad
    x_before = x.grad.copy()
    output_before = output.grad.copy()
    x.data[0] = 99.0

    with pytest.raises(RuntimeError, match="modified after forward"):
        jvp(output, x, np.array([1.0, 1.0]))

    assert x.grad is x_buffer
    assert output.grad is output_buffer
    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(output.grad, output_before)


@pytest.mark.parametrize(
    "tangents, error_type, message",
    [
        (3.0, TypeError, "iterable"),
        ((np.ones(2),), ValueError, "exactly one"),
        ((np.ones(3), np.ones(2)), ValueError, "shape mismatch"),
        ((np.array([True, False]), np.ones(2)), TypeError, "real numeric"),
        ((np.array([np.inf, 0.0]), np.ones(2)), ValueError, "finite"),
    ],
)
def test_jvp_validates_multi_input_tangents(tangents, error_type, message):
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([5.0, 7.0], requires_grad=True)
    output = x * y

    with pytest.raises(error_type, match=message):
        jvp(output, (x, y), tangents)


def test_jvp_validates_single_input_tangent_shape():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x

    with pytest.raises(ValueError, match="tangent 0 shape mismatch"):
        jvp(output, x, np.ones((1, 2)))


def test_jvp_single_element_iterable_rejects_unwrapped_tangent():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x

    with pytest.raises(ValueError, match="exactly one"):
        jvp(output, (x,), np.ones(2))
