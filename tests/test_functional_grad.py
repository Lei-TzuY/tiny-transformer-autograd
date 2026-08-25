"""Functional autograd gradients should be isolated from persistent buffers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine import grad
from engine.tensor import Tensor


def test_grad_returns_scalar_loss_gradient_without_persisting_buffers():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)

    (dx,) = grad(output, x)

    np.testing.assert_allclose(dx, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))
    np.testing.assert_array_equal(output.grad, np.zeros_like(output.data))


def test_grad_supports_explicit_vjp_seed_and_multiple_inputs():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([5.0, 7.0], requires_grad=True)
    output = x * y
    seed = np.array([11.0, 13.0])

    dx, dy = grad(output, (x, y), seed)

    np.testing.assert_allclose(dx, seed * np.asarray(y.data))
    np.testing.assert_allclose(dy, seed * np.asarray(x.data))


def test_grad_restores_preexisting_leaf_and_intermediate_gradients():
    x = Tensor([2.0, 3.0], requires_grad=True)
    hidden = x * x
    output = ops.sum(hidden)
    x.grad[:] = np.array([17.0, 19.0])
    hidden.grad[:] = np.array([23.0, 29.0])
    output.grad[...] = 31.0

    x_before = x.grad.copy()
    hidden_before = hidden.grad.copy()
    output_before = output.grad.copy()

    (dx,) = grad(output, x)

    np.testing.assert_allclose(dx, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(hidden.grad, hidden_before)
    np.testing.assert_array_equal(output.grad, output_before)


def test_grad_returns_independent_arrays():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)

    (dx,) = grad(output, x)
    dx[:] = -1.0

    np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))


def test_grad_rejects_unreachable_input_without_mutating_gradients():
    x = Tensor([2.0], requires_grad=True)
    unused = Tensor([3.0], requires_grad=True)
    output = ops.sum(x * x)
    x.grad[:] = 5.0
    unused.grad[:] = 7.0

    with pytest.raises(ValueError, match="reachable"):
        grad(output, (x, unused))

    np.testing.assert_array_equal(x.grad, np.array([5.0]))
    np.testing.assert_array_equal(unused.grad, np.array([7.0]))


def test_grad_restores_buffers_when_backward_rejects_stale_graph():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = ops.sum(x * x)
    x.grad[:] = np.array([5.0, 7.0])
    output.grad[...] = 11.0
    x_before = x.grad.copy()
    output_before = output.grad.copy()
    x.data[0] = 99.0

    with pytest.raises(RuntimeError, match="modified after forward"):
        grad(output, x)

    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(output.grad, output_before)


@pytest.mark.parametrize(
    "inputs, error",
    [
        ((), "at least one"),
        ((np.array([1.0]),), "only Tensors"),
        ((Tensor([1.0]),), "require gradients"),
    ],
)
def test_grad_validates_inputs(inputs, error):
    x = Tensor([2.0], requires_grad=True)
    output = ops.sum(x * x)

    with pytest.raises((TypeError, ValueError), match=error):
        grad(output, inputs)
