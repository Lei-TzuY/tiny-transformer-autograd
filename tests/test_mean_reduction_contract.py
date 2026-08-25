"""Public contracts for differentiable mean reduction."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.ops import mean
from engine.tensor import Tensor


def test_mean_accepts_numpy_integer_axis_with_correct_vjp():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    out = mean(x, axis=np.int64(1))

    np.testing.assert_allclose(out.data, np.mean(x.data, axis=1))
    out.backward(np.array([2.0, -3.0]))
    np.testing.assert_allclose(
        x.grad,
        np.array([[2.0 / 3.0] * 3, [-1.0] * 3]),
    )


def test_mean_accepts_tuple_with_numpy_and_negative_axes():
    data = np.arange(24.0).reshape(2, 3, 4)
    x = Tensor(data, requires_grad=True)
    out = mean(x, axis=(np.int64(0), -1), keepdims=True)

    np.testing.assert_allclose(out.data, np.mean(data, axis=(0, -1), keepdims=True))
    out.backward(np.ones_like(out.data))
    np.testing.assert_allclose(x.grad, np.full_like(data, 1.0 / 8.0))


@pytest.mark.parametrize("axis", [True, np.bool_(False)])
def test_mean_rejects_boolean_axis(axis):
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    before = x.grad.copy()
    with pytest.raises(TypeError, match="axis"):
        mean(x, axis=axis)
    np.testing.assert_array_equal(x.grad, before)


def test_mean_rejects_empty_full_reduction_explicitly():
    x = Tensor(np.empty((0, 3)), requires_grad=True)
    before = x.grad.copy()
    with pytest.raises(ValueError, match="no elements"):
        mean(x)
    np.testing.assert_array_equal(x.grad, before)


def test_mean_rejects_empty_selected_axis_explicitly():
    x = Tensor(np.empty((2, 0, 3)), requires_grad=True)
    before = x.grad.copy()
    with pytest.raises(ValueError, match="no elements"):
        mean(x, axis=1)
    np.testing.assert_array_equal(x.grad, before)


def test_mean_over_empty_axis_tuple_is_identity():
    data = np.empty((2, 0, 3))
    x = Tensor(data, requires_grad=True)
    out = mean(x, axis=())

    assert out.shape == data.shape
    np.testing.assert_array_equal(out.data, data)
    out.backward(np.empty_like(data))
    np.testing.assert_array_equal(x.grad, np.empty_like(data))
