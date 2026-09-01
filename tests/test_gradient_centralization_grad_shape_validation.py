"""Entry gradient-shape metadata validation for gradient centralization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import centralize_gradients_
from engine.tensor import Tensor


def test_stale_gradient_shape_metadata_fails_before_gradient_write():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = np.array([[1.0, 3.0]], dtype=np.float64)
    parameter.grad = gradient
    parameter._grad_shape = (999,)
    gradient_before = np.array(gradient, copy=True)

    with pytest.raises(ValueError, match="gradient shape metadata"):
        centralize_gradients_([parameter])

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)
    assert parameter._grad_shape == (999,)
