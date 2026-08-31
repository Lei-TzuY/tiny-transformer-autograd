import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptRollbackWriteInput(np.ndarray):
    """Fail the commit, then mutate the value object handed to rollback."""

    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.writes = 0
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.writes = getattr(source, "writes", 0)

    def __setitem__(self, key, value):
        self.writes += 1
        if self.writes == 2:
            value[...] = 0.0
        super().__setitem__(key, value)
        if self.writes == 1:
            raise RuntimeError("injected commit failure")


def test_rollback_write_cannot_redefine_canonical_original_gradient():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = CorruptRollbackWriteInput([6.0, 8.0])
    gradient_ref = parameter.grad
    data_before = parameter.data.copy()
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="adaptive gradient clipping rollback failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, np.zeros(2))
    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter._version == version_before
