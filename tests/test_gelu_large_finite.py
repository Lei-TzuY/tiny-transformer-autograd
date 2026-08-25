"""GELU must remain finite and warning-free for large finite activations."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.ops import gelu
from engine.tensor import Tensor


def _historical_gelu_and_derivative(values):
    """Reproduce the original tanh-GELU statement-level arithmetic exactly."""
    values = np.asarray(values, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    inner = c * (values + 0.044715 * values ** 3)
    t = np.tanh(inner)
    forward = 0.5 * values * (1.0 + t)
    sech2 = 1.0 - t * t
    dtanh_dx = c * (1.0 + 3.0 * 0.044715 * values ** 2)
    derivative = 0.5 * (1.0 + t) + 0.5 * values * sech2 * dtanh_dx
    return forward, derivative


def test_large_finite_gelu_saturates_without_warning_or_nan_gradient():
    values = np.array([1e155, -1e155, 1e200, -1e200], dtype=np.float64)
    x = Tensor(values, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = gelu(x)
        out.backward(np.ones_like(values))

    np.testing.assert_array_equal(
        out.data,
        np.array([1e155, -0.0, 1e200, -0.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(x.grad, np.array([1.0, 0.0, 1.0, 0.0]))
    assert np.isfinite(out.data).all()
    assert np.isfinite(x.grad).all()


def test_large_finite_gelu_respects_upstream_cotangent():
    values = np.array([1e200, -1e200], dtype=np.float64)
    upstream = np.array([2.5, -7.0], dtype=np.float64)
    x = Tensor(values, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gelu(x).backward(upstream)

    np.testing.assert_array_equal(x.grad, np.array([2.5, 0.0]))


def test_large_finite_scalar_gelu_backward_is_stable():
    x = Tensor(np.array(1e200), requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = gelu(x)
        out.backward()

    assert out.data == np.array(1e200)
    assert x.grad == np.array(1.0)


def test_ordinary_range_keeps_historical_gelu_arithmetic_exactly():
    values = np.linspace(-6.0, 6.0, 121, dtype=np.float64)
    expected_forward, expected_derivative = _historical_gelu_and_derivative(values)
    x = Tensor(values, requires_grad=True)

    out = gelu(x)
    out.backward(np.ones_like(values))

    # Saturated elements are masked only before the polynomial derivative;
    # ordinary non-saturated values retain the historical statement grouping.
    np.testing.assert_array_equal(out.data, expected_forward)
    np.testing.assert_array_equal(x.grad, expected_derivative)
