import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutateGradShapeOnWrite(np.ndarray):
    """Gradient storage that corrupts its owner's gradient-shape metadata on writes."""

    def __new__(cls, values, owner):
        result = np.asarray(values, dtype=np.float64).view(cls)
        result._owner = owner
        return result

    def __array_finalize__(self, source):
        self._owner = getattr(source, "_owner", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self._owner is not None:
            self._owner._grad_shape = ("corrupted",)


def test_gradient_write_cannot_corrupt_tensor_grad_shape_metadata():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = MutateGradShapeOnWrite([6.0, 8.0], parameter)
    parameter.grad = gradient
    original_shape = parameter._grad_shape

    with pytest.raises(RuntimeError, match="gradient shape metadata changed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert parameter.grad is gradient
    assert parameter._grad_shape == original_shape
    np.testing.assert_array_equal(np.asarray(gradient), np.array([6.0, 8.0]))


def test_malformed_grad_shape_metadata_fails_before_gradient_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = MutateGradShapeOnWrite([6.0, 8.0], parameter)
    parameter.grad = gradient
    parameter._grad_shape = (1, 2)

    with pytest.raises(ValueError, match="gradient shape metadata"):
        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    np.testing.assert_array_equal(np.asarray(gradient), np.array([6.0, 8.0]))
