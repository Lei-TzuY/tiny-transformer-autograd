import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class ReplaceBackwardOnWrite(np.ndarray):
    def __new__(cls, values, owner):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.owner = owner
        obj.corruptions = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.owner = getattr(source, "owner", None)
            self.corruptions = getattr(source, "corruptions", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.corruptions > 0:
            self.corruptions -= 1
            self.owner._backward_fn = lambda: None


def test_gradient_write_cannot_replace_leaf_backward_closure():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    backward_before = parameter._backward_fn
    gradient = ReplaceBackwardOnWrite([6.0, 8.0], parameter)
    parameter.grad = gradient
    gradient_before = gradient.copy()

    with pytest.raises(RuntimeError, match="backward metadata changed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter._backward_fn is backward_before
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)


def test_malformed_leaf_backward_closure_is_rejected_before_gradient_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0])
    parameter.grad = gradient
    gradient_before = gradient.copy()
    parameter._backward_fn = lambda: None

    with pytest.raises(TypeError, match="parameter 0 backward metadata"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)
