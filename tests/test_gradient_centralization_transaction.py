"""Failure-path regressions for gradient-centralization transactions."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_centralization import centralize_gradients_
from engine.tensor import Tensor


class _MutateThenRaise(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    def __array_finalize__(self, obj):
        pass

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        raise RuntimeError("injected gradient write failure")


def test_mutate_then_raise_destination_is_rolled_back():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _MutateThenRaise([[1.0, 3.0]])
    parameter.grad = gradient
    before = np.array(gradient, copy=True)

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        centralize_gradients_([parameter])

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, before)
