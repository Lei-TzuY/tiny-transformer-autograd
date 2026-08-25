"""Backward cotangents must be real, finite, correctly shaped numeric values."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def _tracked_output():
    x = Tensor([2.0, -3.0], requires_grad=True)
    out = x * x
    return x, out


def test_backward_accepts_real_numeric_array_like_seed():
    x, out = _tracked_output()
    out.backward([2, -1])
    np.testing.assert_allclose(x.grad, np.array([8.0, 6.0]))


@pytest.mark.parametrize(
    "seed",
    [
        np.array([True, False]),
        ["1.0", "2.0"],
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0, 2.0], dtype=object),
    ],
)
def test_backward_rejects_non_real_numeric_seed_types_transactionally(seed):
    x, out = _tracked_output()
    x.grad[:] = np.array([7.0, 11.0])
    before = x.grad.copy()

    with pytest.raises(TypeError, match="gradient.*real numeric"):
        out.backward(seed)

    np.testing.assert_array_equal(x.grad, before)


@pytest.mark.parametrize(
    "seed",
    [
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([-np.inf, 1.0]),
    ],
)
def test_backward_rejects_nonfinite_seed_transactionally(seed):
    x, out = _tracked_output()
    x.grad[:] = np.array([5.0, 13.0])
    before = x.grad.copy()

    with pytest.raises(ValueError, match="gradient.*finite"):
        out.backward(seed)

    np.testing.assert_array_equal(x.grad, before)


def test_backward_shape_failure_remains_transactional():
    x, out = _tracked_output()
    x.grad[:] = np.array([3.0, 4.0])
    before = x.grad.copy()

    with pytest.raises(ValueError, match="shape mismatch"):
        out.backward(np.array([[1.0, 2.0]]))

    np.testing.assert_array_equal(x.grad, before)


def test_invalid_seed_does_not_reset_existing_intermediate_gradient():
    x = Tensor([2.0, 3.0], requires_grad=True)
    hidden = x * x
    out = hidden * 4.0
    out.backward(np.array([1.0, 2.0]))
    hidden_before = hidden.grad.copy()
    leaf_before = x.grad.copy()

    with pytest.raises(ValueError, match="gradient.*finite"):
        out.backward(np.array([np.nan, 1.0]))

    np.testing.assert_array_equal(hidden.grad, hidden_before)
    np.testing.assert_array_equal(x.grad, leaf_before)


def test_scalar_output_accepts_scalar_numpy_seed():
    x = Tensor(3.0, requires_grad=True)
    out = x * x
    out.backward(np.float32(2.0))
    np.testing.assert_allclose(x.grad, 12.0)
