import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptVersionOnRestore(np.ndarray):
    def __new__(cls, values, owner):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.owner = owner
        return array

    def __array_finalize__(self, source):
        self.owner = getattr(source, "owner", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self.owner._version = "corrupted-after-data-rollback"


class MutateParameterThenRaise(np.ndarray):
    def __new__(cls, values, parameter):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.parameter = parameter
        array.write_calls = 0
        return array

    def __array_finalize__(self, source):
        self.parameter = getattr(source, "parameter", None)
        self.write_calls = getattr(source, "write_calls", 0)

    def __setitem__(self, key, value):
        self.write_calls += 1
        np.ndarray.__setitem__(self, key, value)
        if self.write_calls == 1:
            np.ndarray.__setitem__(
                self.parameter.data,
                Ellipsis,
                np.full(self.parameter.data.shape, 7.0),
            )
            raise RuntimeError("injected gradient commit failure")


def test_data_rollback_write_cannot_leave_version_metadata_corrupted():
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    entry_version = parameter._version
    entry_values = np.asarray(parameter.data).copy()

    parameter._data = CorruptVersionOnRestore(entry_values, parameter)
    gradient = MutateParameterThenRaise([100.0, -100.0], parameter)
    parameter.grad = gradient
    entry_gradient = np.asarray(gradient).copy()

    with pytest.raises(RuntimeError, match="injected gradient commit failure"):
        adaptive_clip_grad_([parameter], clip_factor=0.01, eps=1e-3)

    np.testing.assert_array_equal(parameter.data, entry_values)
    np.testing.assert_array_equal(parameter.grad, entry_gradient)
    assert parameter.grad is gradient
    assert parameter._version == entry_version
