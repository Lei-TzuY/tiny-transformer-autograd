import warnings

import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class ReshapeParameterDataOnWrite(np.ndarray):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            np.ndarray.shape.__set__(self.target.data, (1, 2))


def test_parameter_shape_is_restored_after_failed_gradient_commit():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    original_data = parameter.data
    gradient = ReshapeParameterDataOnWrite([6.0, 8.0])
    gradient.target = parameter
    parameter.grad = gradient
    data_before = original_data.copy()
    grad_before = gradient.copy()

    with pytest.raises(RuntimeError, match="parameter data changed for parameter 0"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.data is original_data
    assert parameter.data.shape == (2,)
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
