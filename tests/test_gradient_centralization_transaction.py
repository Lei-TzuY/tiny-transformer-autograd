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


class _RebindOtherGradient(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._target.grad = np.full_like(np.asarray(self._target.data), 99.0)


def test_mutate_then_raise_destination_is_rolled_back():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _MutateThenRaise([[1.0, 3.0]])
    parameter.grad = gradient
    before = np.array(gradient, copy=True)

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        centralize_gradients_([parameter])

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, before)


def test_successful_write_cannot_rebind_another_parameter_gradient():
    victim = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    victim_gradient = np.array([[2.0, 6.0]], dtype=np.float64)
    victim.grad = victim_gradient

    attacker = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    attacker_gradient = _RebindOtherGradient([[1.0, 3.0]], victim)
    attacker.grad = attacker_gradient
    attacker_before = np.array(attacker_gradient, copy=True)
    victim_before = np.array(victim_gradient, copy=True)

    with pytest.raises(RuntimeError, match="gradient binding changed"):
        centralize_gradients_([attacker, victim])

    assert attacker.grad is attacker_gradient
    assert victim.grad is victim_gradient
    np.testing.assert_array_equal(attacker.grad, attacker_before)
    np.testing.assert_array_equal(victim.grad, victim_before)
