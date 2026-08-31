import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class ChangeOwnDtypeOnWrite(np.ndarray):
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
        state = (1, self.shape, np.dtype(np.int64), False, self.tobytes())
        self.__setstate__(state)


def test_commit_detects_and_restores_gradient_dtype_change():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    gradient = ChangeOwnDtypeOnWrite([1.0, -1.0])
    parameter.grad = gradient
    grad_before = gradient.copy()

    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    with pytest.raises(RuntimeError, match="adaptive gradient clipping write failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.5, eps=tiny)

    assert parameter.grad is gradient
    assert parameter.grad.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(parameter.grad, grad_before)
