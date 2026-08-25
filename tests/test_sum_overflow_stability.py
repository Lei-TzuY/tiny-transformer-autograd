"""Recover finite sum reductions without changing ordinary arithmetic."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import ops
from engine.tensor import Tensor


def _float64_bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def test_sum_recovers_finite_result_after_intermediate_overflow():
    x = Tensor([1e308, 1e308, -1e308], requires_grad=True)

    with np.errstate(all="raise"):
        out = ops.sum(x)

    assert out.data == np.float64(1e308)
    out.backward()
    np.testing.assert_array_equal(x.grad, np.ones(3))


def test_sum_recovers_negative_finite_result_after_intermediate_overflow():
    x = Tensor([-1e308, -1e308, 1e308], requires_grad=True)

    with np.errstate(all="raise"):
        out = ops.sum(x)

    assert out.data == np.float64(-1e308)
    out.backward()
    np.testing.assert_array_equal(x.grad, np.ones(3))


def test_mixed_sum_keeps_safe_slice_historical_bits():
    data = np.array(
        [
            [1e308, 1e308, -1e308],
            [1.0, np.nextafter(1.0, 2.0), -0.25],
        ]
    )
    historical_safe = data[1].sum()
    x = Tensor(data, requires_grad=True)

    with np.errstate(all="raise"):
        out = ops.sum(x, axis=1)

    assert out.data[0] == np.float64(1e308)
    assert _float64_bits(out.data[1]) == _float64_bits(historical_safe)

    out.backward(np.array([2.0, -3.0]))
    np.testing.assert_array_equal(
        x.grad,
        np.array([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0]]),
    )


def test_broadcast_backward_recovers_finite_cotangent_sum():
    parent = Tensor([2.0], requires_grad=True)
    offsets = Tensor([10.0, 20.0, 30.0])
    out = ops.add(parent, offsets)

    with np.errstate(all="raise"):
        out.backward(np.array([1e308, 1e308, -1e308]))

    np.testing.assert_array_equal(parent.grad, np.array([1e308]))


def test_broadcast_backward_keeps_ordinary_reduction_bits():
    parent = Tensor([2.0], requires_grad=True)
    offsets = Tensor([10.0, 20.0, 30.0])
    out = ops.add(parent, offsets)
    seed = np.array([1.0, np.nextafter(1.0, 2.0), -0.25])
    historical = seed.sum(keepdims=True)

    out.backward(seed)

    assert _float64_bits(parent.grad[0]) == _float64_bits(historical[0])


def test_sum_preserves_warning_for_genuinely_unrepresentable_result():
    x = Tensor([1e308, 1e308])

    with pytest.warns(RuntimeWarning, match="overflow"):
        out = ops.sum(x)

    assert np.isposinf(out.data)
