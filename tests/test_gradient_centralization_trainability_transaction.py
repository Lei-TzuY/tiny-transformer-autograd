"""Trainability-metadata regressions for gradient-centralization transactions."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_centralization import centralize_gradients_
from engine.tensor import Tensor


class _FreezeTargetOnWrite(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._target.requires_grad = False


def test_gradient_write_cannot_change_parameter_trainability():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _FreezeTargetOnWrite([[1.0, 3.0]], parameter)
    parameter.grad = gradient
    gradient_before = np.array(gradient, copy=True)

    with pytest.raises(RuntimeError, match="trainability changed"):
        centralize_gradients_([parameter])

    assert parameter.requires_grad is True
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)
