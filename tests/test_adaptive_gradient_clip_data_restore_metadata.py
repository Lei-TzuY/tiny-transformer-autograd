import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptVersionOnDataRestore(np.ndarray):
    def __new__(cls, values, owner):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.owner = owner
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.owner = getattr(source, "owner", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.owner._version = "corrupted-after-data-restore"


class MutateParameterDataThenRaise(np.ndarray):
    def __new__(cls, values, owner):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.owner = owner
        obj.failures = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.owner = getattr(source, "owner", None)
            self.failures = getattr(source, "failures", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.failures > 0:
            self.failures -= 1
            np.asarray(self.owner.data)[...] = 99.0
            raise RuntimeError("injected gradient write failure")


def test_data_restore_cannot_leave_parameter_version_metadata_corrupted():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter._data = CorruptVersionOnDataRestore(parameter.data, parameter)
    parameter.grad = MutateParameterDataThenRaise([6.0, 8.0], parameter)
    data_ref = parameter.data
    data_before = parameter.data.copy()
    gradient_ref = parameter.grad
    gradient_before = parameter.grad.copy()
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.data is data_ref
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, gradient_before)
    assert type(parameter._version) is int
    assert parameter._version >= version_before
