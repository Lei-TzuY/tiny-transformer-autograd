"""Focused correctness tests for Tensor scalar power."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def test_zero_exponent_has_zero_gradient_at_zero_without_warnings():
    x = Tensor(np.array([0.0, 2.0, -3.0]), requires_grad=True)
    seed = np.array([2.0, -1.5, 4.0])

    with np.errstate(all="raise"):
        y = x ** 0
        y.backward(seed)

    np.testing.assert_array_equal(y.data, np.ones(3))
    np.testing.assert_array_equal(x.grad, np.zeros(3))


def test_numpy_real_scalar_exponents_are_supported():
    x = Tensor(np.array([0.5, 1.5, 2.0]), requires_grad=True)
    exponent = np.float64(3.0)
    seed = np.array([0.25, -2.0, 1.5])

    (x ** exponent).backward(seed)
    expected = seed * 3.0 * x.data ** 2
    np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    z = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (z ** np.int64(2)).backward(np.array([3.0, 4.0]))
    np.testing.assert_allclose(z.grad, np.array([6.0, 16.0]), atol=1e-12)


@pytest.mark.parametrize("exponent", [True, False, np.bool_(True), 1 + 2j, "2"])
def test_power_rejects_non_real_scalar_exponents(exponent):
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(TypeError, match="real scalar"):
        _ = x ** exponent


@pytest.mark.parametrize("exponent", [np.nan, np.inf, -np.inf, np.float64(np.nan)])
def test_power_rejects_non_finite_exponents(exponent):
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        _ = x ** exponent


def test_zero_power_backward_allocates_zero_grad_for_lazy_leaf():
    x = Tensor([0.0, 4.0], requires_grad=True)
    x.grad = None

    (x ** 0.0).backward(np.array([5.0, 7.0]))

    assert x.grad is not None
    np.testing.assert_array_equal(x.grad, np.zeros(2))
