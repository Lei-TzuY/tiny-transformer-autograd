"""Mean reductions should not overflow before a representable division."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import engine.ops as ops


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def test_mean_recovers_same_sign_large_finite_values_without_warning():
    x = Tensor(np.array([1e308, 1e308]), requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ops.mean(x)
        result.backward()

    assert result.data == np.float64(1e308)
    np.testing.assert_array_equal(x.grad, np.array([0.5, 0.5]))


def test_mean_recovers_large_finite_cancellation_without_warning():
    x = Tensor(
        np.array([1e308, 1e308, -1e308, -1e308]),
        requires_grad=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ops.mean(x)
        result.backward()

    assert result.data == 0.0
    np.testing.assert_array_equal(x.grad, np.full(4, 0.25))


def test_mean_recovers_negative_large_finite_values():
    x = Tensor(np.array([-1e308, -1e308]), requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ops.mean(x)
        result.backward()

    assert result.data == np.float64(-1e308)
    np.testing.assert_array_equal(x.grad, np.array([0.5, 0.5]))


def test_mean_multi_axis_keepdims_recovers_large_finite_values():
    x = Tensor(np.full((2, 2, 2), 1e308), requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ops.mean(x, axis=(1, 2), keepdims=True)
        result.backward()

    assert result.shape == (2, 1, 1)
    np.testing.assert_array_equal(result.data, np.full((2, 1, 1), 1e308))
    np.testing.assert_array_equal(x.grad, np.full((2, 2, 2), 0.25))


def test_mixed_reduction_preserves_safe_row_historical_bits_and_vjp():
    # This row is intentionally chosen so sum(row)/3 and sum(row/3) differ by
    # one float64 ULP. It proves an overflowing neighbour does not move a safe
    # row onto the fallback arithmetic path.
    ordinary = np.array(
        [
            0.014037549709959801,
            0.005481572330864507,
            -0.0015754414115104593,
        ]
    )
    historical = ordinary.sum() * (1.0 / 3.0)
    early_scaled = (ordinary * (1.0 / 3.0)).sum()
    assert _bits(historical) != _bits(early_scaled)

    data = np.vstack([np.full(3, 1e308), ordinary])
    x = Tensor(data, requires_grad=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ops.mean(x, axis=1)
        result.backward()

    assert result.data[0] == np.float64(1e308)
    assert _bits(result.data[1]) == _bits(historical)
    np.testing.assert_array_equal(x.grad, np.full_like(data, 1.0 / 3.0))


def test_nonfinite_source_is_not_reclassified_as_recoverable_overflow():
    values = Tensor(np.array([[np.inf, 1.0], [np.nan, 2.0]]))

    result = ops.mean(values, axis=1)

    assert np.isposinf(result.data[0])
    assert np.isnan(result.data[1])
