import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class ReshapeSelfOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.reshape_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.reshape_remaining = getattr(source, "reshape_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.reshape_remaining <= 0:
            return
        self.reshape_remaining -= 1
        np.ndarray.shape.__set__(self, (1, self.size))


def test_failed_commit_restores_exact_gradient_shape_on_same_object():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = ReshapeSelfOnWrite([6.0, 8.0])
    parameter.grad = gradient
    grad_before = gradient.copy()

    with pytest.raises(RuntimeError, match="adaptive gradient clipping write failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    assert gradient.shape == (2,)
    np.testing.assert_array_equal(gradient, grad_before)
