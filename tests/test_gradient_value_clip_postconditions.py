import numpy as np
import pytest

from engine.gradient_value_clip import clip_grad_value_
from engine.tensor import Tensor


class _IgnoreFirstWrite(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.ignore_writes = 1
        return array

    def __array_finalize__(self, source):
        self.ignore_writes = getattr(source, "ignore_writes", 0)

    def __setitem__(self, key, value):
        if self.ignore_writes:
            self.ignore_writes -= 1
            return
        super().__setitem__(key, value)


class _CorruptFirstWrite(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.corrupt_writes = 1
        return array

    def __array_finalize__(self, source):
        self.corrupt_writes = getattr(source, "corrupt_writes", 0)

    def __setitem__(self, key, value):
        if self.corrupt_writes:
            self.corrupt_writes -= 1
            super().__setitem__(key, np.zeros_like(np.asarray(value)))
            return
        super().__setitem__(key, value)


class _CorruptThenDropRollback(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.write_count = 0
        return array

    def __array_finalize__(self, source):
        self.write_count = getattr(source, "write_count", 0)

    def __setitem__(self, key, value):
        self.write_count += 1
        if self.write_count == 1:
            super().__setitem__(key, np.zeros_like(np.asarray(value)))
        # The second write is rollback; silently ignore it so verification must fail.


def _parameter_with_gradient(gradient):
    parameter = Tensor(np.zeros(np.asarray(gradient).shape), requires_grad=True)
    parameter.grad = gradient
    return parameter


def test_silently_ignored_commit_is_detected_and_state_is_restored():
    gradient = _IgnoreFirstWrite([9.0, -8.0])
    parameter = _parameter_with_gradient(gradient)
    before = gradient.copy()

    with pytest.raises(RuntimeError, match="commit did not store requested values"):
        clip_grad_value_(parameter, 1.0)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)


def test_silently_corrupted_commit_is_detected_and_rolled_back():
    gradient = _CorruptFirstWrite([9.0, -8.0])
    parameter = _parameter_with_gradient(gradient)
    before = gradient.copy()

    with pytest.raises(RuntimeError, match="commit did not store requested values"):
        clip_grad_value_(parameter, 1.0)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)


def test_silent_rollback_failure_is_detected_explicitly():
    gradient = _CorruptThenDropRollback([9.0, -8.0])
    parameter = _parameter_with_gradient(gradient)

    with pytest.raises(RuntimeError, match="gradient value clipping rollback failed"):
        clip_grad_value_(parameter, 1.0)

    np.testing.assert_array_equal(gradient, np.zeros(2))
