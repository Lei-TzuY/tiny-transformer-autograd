import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CountingWrites(np.ndarray):
    def __new__(cls, values):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array.write_count = 0
        return array

    def __array_finalize__(self, source):
        self.write_count = getattr(source, "write_count", 0)

    def __setitem__(self, key, value):
        self.write_count += 1
        super().__setitem__(key, value)


def test_parameter_data_is_validated_even_when_gradient_is_none():
    parameter = Tensor([1.0], requires_grad=True)
    parameter._data = np.array([np.nan])

    with pytest.raises(ValueError, match=r"parameter 0 data.*finite"):
        adaptive_clip_grad_(parameter)


def test_inactive_parameter_data_requires_real_numeric_storage():
    parameter = Tensor([1.0], requires_grad=True)
    parameter._data = np.array([object()], dtype=object)

    with pytest.raises(TypeError, match=r"parameter 0 data.*real numeric dtype"):
        adaptive_clip_grad_(parameter)


def test_late_invalid_inactive_parameter_fails_before_earlier_gradient_write():
    first = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = CountingWrites([6.0, 8.0])
    second = Tensor([1.0], requires_grad=True)
    second._data = np.array([np.nan])
    first_before = np.array(first.grad, copy=True)

    with pytest.raises(ValueError, match=r"parameter 1 data.*finite"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    assert first.grad.write_count == 0
    np.testing.assert_array_equal(first.grad, first_before)
