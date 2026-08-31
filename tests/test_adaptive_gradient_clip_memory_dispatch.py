import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class LyingSharesMemory(np.ndarray):
    shares_memory_calls = 0
    write_calls = 0

    def __array_function__(self, func, types, args, kwargs):
        if func is np.shares_memory:
            type(self).shares_memory_calls += 1
            return False
        return super().__array_function__(func, types, args, kwargs)

    def __setitem__(self, key, value):
        type(self).write_calls += 1
        return super().__setitem__(key, value)


def _reset_counters():
    LyingSharesMemory.shares_memory_calls = 0
    LyingSharesMemory.write_calls = 0


def test_overlap_preflight_does_not_trust_gradient_array_function_dispatch():
    _reset_counters()
    storage = np.array([6.0, 8.0, 10.0], dtype=np.float64).view(LyingSharesMemory)
    left = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    right = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    left.grad = storage[:2]
    right.grad = storage[1:]
    original = storage.copy()

    with pytest.raises(ValueError, match="gradient storage must not overlap"):
        adaptive_clip_grad_([left, right], clip_factor=0.1, eps=1e-3)

    np.testing.assert_array_equal(storage, original)
    assert LyingSharesMemory.shares_memory_calls == 0
    assert LyingSharesMemory.write_calls == 0


def test_parameter_alias_preflight_does_not_trust_gradient_dispatch():
    _reset_counters()
    parameter = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    parameter.grad = parameter.data.view(LyingSharesMemory)
    original_data = parameter.data.copy()
    original_version = parameter._version

    with pytest.raises(ValueError, match="must not overlap parameter 0 data"):
        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    np.testing.assert_array_equal(parameter.data, original_data)
    assert parameter._version == original_version
    assert LyingSharesMemory.shares_memory_calls == 0
    assert LyingSharesMemory.write_calls == 0
