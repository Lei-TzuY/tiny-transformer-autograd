"""Division denominator gradients must survive extreme finite intermediates."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def _denominator_grad(numerator, denominator, upstream):
    numerator = Tensor(np.asarray(numerator, dtype=np.float64))
    denominator = Tensor(np.asarray(denominator, dtype=np.float64), requires_grad=True)
    quotient = numerator / denominator
    quotient.backward(np.asarray(upstream, dtype=np.float64))
    return quotient.data, denominator.grad


def test_denominator_square_overflow_does_not_zero_representable_gradient():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient, grad = _denominator_grad(1e308, 1e200, 1.0)

    assert quotient == np.array(1e108)
    np.testing.assert_allclose(grad, -1e-92, rtol=2e-15, atol=0.0)
    assert grad != 0.0


def test_overflowing_numerator_and_square_cancel_to_finite_gradient():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient, grad = _denominator_grad(1e200, 1e200, 1e200)

    assert quotient == np.array(1.0)
    assert grad == np.array(-1.0)


def test_underflowing_numerator_keeps_representable_gradient():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient, grad = _denominator_grad(1e-200, 1e-100, 1e-200)

    assert quotient == np.array(1e-100)
    np.testing.assert_allclose(grad, -1e-200, rtol=2e-15, atol=0.0)
    assert grad != 0.0


def test_underflowing_numerator_and_square_cancel_to_finite_gradient():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient, grad = _denominator_grad(1e-200, 1e-200, 1e-200)

    assert quotient == np.array(1.0)
    assert grad == np.array(-1.0)


def test_negative_extreme_denominator_keeps_squared_derivative_sign():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient, grad = _denominator_grad(1e308, -1e200, 1.0)

    assert quotient == np.array(-1e108)
    np.testing.assert_allclose(grad, -1e-92, rtol=2e-15, atol=0.0)


def test_extreme_broadcast_gradients_are_summed_after_stable_elementwise_vjp():
    numerator = np.array(
        [
            [1e308, 1e-200, 1e308],
            [5e307, 5e-201, 5e307],
        ]
    )
    denominator = Tensor(
        np.array([1e200, 1e-200, -1e200]),
        requires_grad=True,
    )
    upstream = np.array(
        [
            [1.0, 1e-200, 1.0],
            [2.0, 2e-200, 2.0],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (Tensor(numerator) / denominator).backward(upstream)

    np.testing.assert_allclose(
        denominator.grad,
        np.array([-2e-92, -2.0, -2e-92]),
        rtol=2e-15,
        atol=0.0,
    )


def test_ordinary_denominator_gradient_keeps_historical_arithmetic_exactly():
    rng = np.random.default_rng(20260825)
    numerator = rng.uniform(-3.0, 3.0, size=(7, 11))
    denominator_data = rng.uniform(0.5, 3.0, size=(7, 11))
    denominator_data *= rng.choice(np.array([-1.0, 1.0]), size=(7, 11))
    upstream = rng.uniform(-2.0, 2.0, size=(7, 11))

    denominator = Tensor(denominator_data, requires_grad=True)
    (Tensor(numerator) / denominator).backward(upstream)

    expected = -upstream * numerator / (denominator_data * denominator_data)
    np.testing.assert_array_equal(denominator.grad, expected)
