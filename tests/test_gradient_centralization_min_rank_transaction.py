"""Transaction regressions for live gradients excluded only by ``min_rank``."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import Tensor, centralize_gradients_


class _MutateLowerRankGradient(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        np.ndarray.__setitem__(self._target.grad, Ellipsis, 99.0)


def test_changed_gradient_cannot_mutate_valid_gradient_excluded_by_min_rank():
    victim = Tensor(np.zeros((2,), dtype=np.float64), requires_grad=True)
    victim_gradient = np.array([2.0, 6.0], dtype=np.float64)
    victim.grad = victim_gradient

    attacker = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    attacker_gradient = _MutateLowerRankGradient([[1.0, 3.0]], victim)
    attacker.grad = attacker_gradient

    attacker_before = np.array(attacker_gradient, copy=True)
    victim_before = np.array(victim_gradient, copy=True)

    with pytest.raises(RuntimeError, match="gradient value changed"):
        centralize_gradients_([attacker, victim])

    assert attacker.grad is attacker_gradient
    assert victim.grad is victim_gradient
    np.testing.assert_array_equal(attacker.grad, attacker_before)
    np.testing.assert_array_equal(victim.grad, victim_before)
