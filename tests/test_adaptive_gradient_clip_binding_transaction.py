import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class RebindGradOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.replacement = None
        obj.rebinds_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.replacement = getattr(source, "replacement", None)
            self.rebinds_remaining = getattr(source, "rebinds_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.rebinds_remaining > 0:
            self.rebinds_remaining -= 1
            self.target.grad = self.replacement


class ChangeTrainabilityOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.replacement = False
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.replacement = getattr(source, "replacement", False)
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining > 0:
            self.changes_remaining -= 1
            self.target.requires_grad = self.replacement


def test_commit_detects_and_restores_own_gradient_binding_replacement():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    original = RebindGradOnWrite([6.0, 8.0])
    replacement = np.array([-1.0, -2.0])
    original.target = parameter
    original.replacement = replacement
    parameter.grad = original
    before = original.copy()

    with pytest.raises(RuntimeError, match="gradient binding changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is original
    np.testing.assert_array_equal(parameter.grad, before)
    np.testing.assert_array_equal(replacement, [-1.0, -2.0])


def test_commit_detects_cross_parameter_binding_replacement_before_later_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first_gradient = RebindGradOnWrite([6.0, 8.0])
    second_gradient = np.array([6.0, 8.0])
    replacement = np.array([-9.0, -9.0])
    first_gradient.target = second
    first_gradient.replacement = replacement
    first.grad = first_gradient
    second.grad = second_gradient
    first_before = first_gradient.copy()
    second_before = second_gradient.copy()

    with pytest.raises(RuntimeError, match="gradient binding changed for parameter 1"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    assert first.grad is first_gradient
    assert second.grad is second_gradient
    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)
    np.testing.assert_array_equal(replacement, [-9.0, -9.0])


def test_commit_detects_and_restores_own_trainability_change():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = ChangeTrainabilityOnWrite([6.0, 8.0])
    gradient.target = parameter
    parameter.grad = gradient
    before = gradient.copy()

    with pytest.raises(RuntimeError, match="trainability changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.requires_grad is True
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, before)


def test_cross_parameter_trainability_change_is_detected_before_later_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first_gradient = ChangeTrainabilityOnWrite([6.0, 8.0])
    second_gradient = np.array([6.0, 8.0])
    first_gradient.target = second
    first.grad = first_gradient
    second.grad = second_gradient
    first_before = first_gradient.copy()
    second_before = second_gradient.copy()

    with pytest.raises(RuntimeError, match="trainability changed for parameter 1"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    assert first.requires_grad is True
    assert second.requires_grad is True
    assert first.grad is first_gradient
    assert second.grad is second_gradient
    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)
