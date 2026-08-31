import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptAssignedPayload(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    def __array_finalize__(self, source):
        pass

    def __setitem__(self, key, value):
        payload = np.asarray(value)
        payload[...] = 0.0
        super().__setitem__(key, payload)


class MutateParameterDataOnGradientWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining <= 0:
            return
        self.changes_remaining -= 1
        np.asarray(self.target.data)[...] = [91.0, 92.0]


def test_parameter_rollback_keeps_canonical_entry_snapshot_private():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter._data = CorruptAssignedPayload([3.0, 4.0])
    gradient = MutateParameterDataOnGradientWrite([6.0, 8.0])
    gradient.target = parameter
    parameter.grad = gradient
    grad_before = gradient.copy()

    with pytest.raises(RuntimeError, match="adaptive gradient clipping rollback failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.data is not None
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
