"""Gradient-shape metadata regressions for centralization transactions."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import centralize_gradients_
from engine.tensor import Tensor


class _CorruptGradShapeOnWrite(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._target._grad_shape = (999,)


def test_gradient_write_cannot_change_tensor_grad_shape_metadata():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _CorruptGradShapeOnWrite([[1.0, 3.0]], parameter)
    parameter.grad = gradient
    gradient_before = np.array(gradient, copy=True)
    grad_shape_before = parameter._grad_shape

    with pytest.raises(RuntimeError, match="gradient shape metadata changed"):
        centralize_gradients_([parameter])

    assert parameter._grad_shape == grad_shape_before
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)
