import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def test_malformed_requires_grad_is_rejected_even_without_a_gradient():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.requires_grad = np.bool_(True)

    with pytest.raises(TypeError, match="parameter 0 requires_grad must be a bool"):
        adaptive_clip_grad_(parameter)


def test_truthy_non_boolean_requires_grad_cannot_enable_gradient_writes():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0])
    parameter.requires_grad = "trainable"
    before = parameter.grad.copy()

    with pytest.raises(TypeError, match="parameter 0 requires_grad must be a bool"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    np.testing.assert_array_equal(parameter.grad, before)


def test_late_malformed_requires_grad_is_rejected_before_earlier_gradient_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([1.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.requires_grad = 1
    first_before = first.grad.copy()

    with pytest.raises(TypeError, match="parameter 1 requires_grad must be a bool"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    np.testing.assert_array_equal(first.grad, first_before)
