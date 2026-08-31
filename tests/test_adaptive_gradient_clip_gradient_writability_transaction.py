import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MakeSelfReadOnlyOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining <= 0:
            return
        self.changes_remaining -= 1
        np.ndarray.setflags(self, write=False)


def test_commit_detects_and_restores_gradient_writability_change():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = MakeSelfReadOnlyOnWrite([6.0, 8.0])
    parameter.grad = gradient
    grad_before = gradient.copy()

    assert np.asarray(gradient).flags.writeable

    with pytest.raises(RuntimeError, match="gradient writability changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    assert np.asarray(gradient).flags.writeable
    np.testing.assert_array_equal(gradient, grad_before)
