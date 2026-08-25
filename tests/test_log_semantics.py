"""Focused semantic tests for the natural-log autograd primitive."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.tensor import Tensor


def test_log_matches_numpy_for_tiny_positive_values_and_vjp():
    values = np.array([1e-300, 1e-20, 1e-6, 1.0, np.e])
    upstream = np.array([0.5, -2.0, 3.0, 4.0, -1.5])
    x = Tensor(values, requires_grad=True)

    y = ops.log(x)
    y.backward(upstream)

    np.testing.assert_array_equal(y.data, np.log(values))
    np.testing.assert_allclose(
        x.grad,
        upstream / values,
        rtol=1e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0],
        [-1.0],
        [1.0, 0.0],
        [np.nan],
        [-np.inf],
    ],
)
def test_log_rejects_values_outside_its_real_positive_domain(values):
    x = Tensor(values, requires_grad=True)
    with pytest.raises(ValueError, match="positive"):
        ops.log(x)

    # A failed forward must not create or mutate gradient state.
    np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))


def test_log_keeps_numpy_positive_infinity_semantics():
    x = Tensor([np.inf], requires_grad=True)
    y = ops.log(x)
    y.backward(np.array([3.0]))

    assert np.isposinf(y.data[0])
    np.testing.assert_array_equal(x.grad, np.array([0.0]))
