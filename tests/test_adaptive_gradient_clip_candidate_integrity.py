import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptFirstWriteInput(np.ndarray):
    """Mutate the value object handed to __setitem__ before storing it."""

    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.corruptions = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.corruptions = getattr(source, "corruptions", 0)

    def __setitem__(self, key, value):
        if self.corruptions > 0:
            self.corruptions -= 1
            value[...] = 0.0
        super().__setitem__(key, value)


def test_gradient_write_cannot_corrupt_internal_candidate_undetected():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = CorruptFirstWriteInput([6.0, 8.0])
    gradient_ref = parameter.grad
    gradient_before = parameter.grad.copy()
    data_before = parameter.data.copy()
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="write failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, gradient_before)
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter._version == version_before
