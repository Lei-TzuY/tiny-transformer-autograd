import warnings

import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def _set_strides(array, strides):
    # Newer NumPy warns on direct layout metadata assignment. This test needs the
    # hostile state itself; warning policy is exercised by the production call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.strides.__set__(array, strides)


class ChangeOwnStridesOnWrite(np.ndarray):
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
        _set_strides(self, (0,))


class ChangeParameterStridesOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining <= 0:
            return
        self.changes_remaining -= 1
        _set_strides(self.target.data, (0,))


def test_commit_detects_and_restores_gradient_strides():
    parameter = Tensor([3.0, 3.0], requires_grad=True)
    gradient = ChangeOwnStridesOnWrite([8.0, 8.0])
    parameter.grad = gradient
    grad_before = gradient.copy()
    strides_before = gradient.strides

    with pytest.raises(RuntimeError, match="gradient strides changed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    assert gradient.strides == strides_before
    np.testing.assert_array_equal(gradient, grad_before)


def test_commit_detects_and_restores_parameter_data_strides():
    parameter = Tensor([3.0, 3.0], requires_grad=True)
    gradient = ChangeParameterStridesOnWrite([8.0, 8.0])
    gradient.target = parameter
    parameter.grad = gradient
    data = parameter.data
    data_before = data.copy()
    strides_before = data.strides
    grad_before = gradient.copy()

    with pytest.raises(RuntimeError, match="parameter data strides changed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.data is data
    assert parameter.data.strides == strides_before
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
