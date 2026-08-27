"""Dense functional Jacobians should be correct and side-effect free."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine import grad, jacobian
from engine.tensor import Tensor


def test_jacobian_scalar_output_has_input_shape():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)

    (dx,) = jacobian(output, x)

    assert dx.shape == x.shape
    np.testing.assert_allclose(dx, np.array([4.0, 6.0]))


def test_jacobian_vector_output_uses_output_then_input_axes():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x + x

    (dx,) = jacobian(output, x)

    assert dx.shape == output.shape + x.shape
    np.testing.assert_allclose(dx, np.diag([5.0, 7.0]))


def test_jacobian_supports_multiple_inputs():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([5.0, 7.0], requires_grad=True)
    output = x * y

    dx, dy = jacobian(output, (x, y))

    np.testing.assert_allclose(dx, np.diag(np.asarray(y.data)))
    np.testing.assert_allclose(dy, np.diag(np.asarray(x.data)))


def test_jacobian_higher_rank_input_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = Tensor([5.0, 7.0])
    output = x @ weights

    (dx,) = jacobian(output, x)

    expected = np.zeros((2, 2, 2), dtype=np.float64)
    expected[0, 0] = np.array([5.0, 7.0])
    expected[1, 1] = np.array([5.0, 7.0])
    assert dx.shape == output.shape + x.shape
    np.testing.assert_allclose(dx, expected)


def test_jacobian_contracts_back_to_grad_vjp():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x + x
    seed = np.array([11.0, 13.0])

    (full,) = jacobian(output, x)
    (vjp,) = grad(output, x, seed)

    np.testing.assert_allclose(seed @ full, vjp)


def test_jacobian_preserves_existing_gradient_buffer_identity_and_values():
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

    jacobian(output, x)

    assert x.grad is x_buffer
    assert hidden.grad is hidden_buffer
    assert output.grad is output_buffer
    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(hidden.grad, hidden_before)
    np.testing.assert_array_equal(output.grad, output_before)


def test_jacobian_duplicate_inputs_return_independent_arrays():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x

    first, second = jacobian(output, (x, x))

    np.testing.assert_array_equal(first, second)
    assert first is not second
    first[:] = -1.0
    np.testing.assert_allclose(second, np.diag([4.0, 6.0]))


def test_jacobian_empty_output_preserves_shape_and_validates_graph():
    x = Tensor([2.0, 3.0, 5.0], requires_grad=True)
    output = x[:0]

    (dx,) = jacobian(output, x)

    assert dx.shape == (0, 3)
    assert dx.dtype == np.float64


def test_jacobian_rejects_unreachable_input_without_mutating_gradients():
    x = Tensor([2.0], requires_grad=True)
    unused = Tensor([3.0], requires_grad=True)
    output = x * x
    x.grad[:] = 5.0
    unused.grad[:] = 7.0

    with pytest.raises(ValueError, match="reachable"):
        jacobian(output, (x, unused))

    np.testing.assert_array_equal(x.grad, np.array([5.0]))
    np.testing.assert_array_equal(unused.grad, np.array([7.0]))


def test_jacobian_restores_buffers_when_backward_rejects_stale_graph():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x
    x.grad[:] = np.array([5.0, 7.0])
    output.grad[:] = np.array([11.0, 13.0])
    x_buffer = x.grad
    output_buffer = output.grad
    x_before = x.grad.copy()
    output_before = output.grad.copy()
    x.data[0] = 99.0

    with pytest.raises(RuntimeError, match="modified after forward"):
        jacobian(output, x)

    assert x.grad is x_buffer
    assert output.grad is output_buffer
    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(output.grad, output_before)
