import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class ChangeParameterDataOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.mode = "mutate"
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.mode = getattr(source, "mode", "mutate")
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining <= 0:
            return
        self.changes_remaining -= 1
        if self.mode == "replace":
            self.target.data = np.full(self.target.shape, 91.0)
        elif self.mode == "version_only":
            self.target.data[...] = self.target.data
        else:
            self.target.data[...] = self.target.data + 7.0


class ExplodingOwnerRef:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise RuntimeError("owner metadata callable must not run")


def test_commit_detects_and_restores_parameter_data_mutation():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = ChangeParameterDataOnWrite([6.0, 8.0])
    gradient.target = parameter
    parameter.grad = gradient
    data_before = parameter.data.copy()
    grad_before = gradient.copy()
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="parameter data changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    np.testing.assert_array_equal(parameter.data, data_before)
    np.testing.assert_array_equal(parameter.grad, grad_before)
    assert parameter.grad is gradient
    assert parameter._version > version_before


def test_commit_restores_replaced_parameter_data_binding():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    original_data = parameter.data
    gradient = ChangeParameterDataOnWrite([6.0, 8.0])
    gradient.target = parameter
    gradient.mode = "replace"
    parameter.grad = gradient
    data_before = original_data.copy()
    grad_before = gradient.copy()

    with pytest.raises(RuntimeError, match="parameter data binding changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.data is original_data
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)


def test_cross_parameter_version_change_is_detected_before_later_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first_gradient = ChangeParameterDataOnWrite([6.0, 8.0])
    second_gradient = np.array([6.0, 8.0])
    first_gradient.target = second
    first_gradient.mode = "version_only"
    first.grad = first_gradient
    second.grad = second_gradient
    first_grad_before = first_gradient.copy()
    second_grad_before = second_gradient.copy()
    second_data_before = second.data.copy()
    second_version_before = second._version

    with pytest.raises(RuntimeError, match="parameter version changed for parameter 1"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    np.testing.assert_array_equal(first.grad, first_grad_before)
    np.testing.assert_array_equal(second.grad, second_grad_before)
    np.testing.assert_array_equal(second.data, second_data_before)
    assert second._version > second_version_before


def test_foreign_tensor_storage_is_rejected_before_gradient_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    foreign = Tensor([9.0, 10.0], requires_grad=True)
    gradient = np.array([6.0, 8.0])
    parameter.grad = gradient
    parameter._data = foreign.data
    grad_before = gradient.copy()
    foreign_version_before = foreign._version

    with pytest.raises(TypeError, match="parameter 0 data must be Tensor-managed storage"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
    assert foreign._version == foreign_version_before


def test_corrupt_owner_metadata_is_rejected_without_callable_dispatch():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0])
    parameter.grad = gradient
    owner_ref = ExplodingOwnerRef()
    parameter.data._owner_ref = owner_ref
    grad_before = gradient.copy()

    with pytest.raises(TypeError, match="parameter 0 data must be Tensor-managed storage"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert owner_ref.calls == 0
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
