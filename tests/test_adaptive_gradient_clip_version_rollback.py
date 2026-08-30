import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptVersionThenRaise(np.ndarray):
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
            self.owner._version = "corrupted"
            raise RuntimeError("injected gradient write failure")


def test_failed_gradient_write_repairs_malformed_parameter_version_metadata():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = CorruptVersionThenRaise([6.0, 8.0], parameter)
    gradient_ref = parameter.grad
    gradient_before = parameter.grad.copy()
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, gradient_before)
    assert type(parameter._version) is int
    assert parameter._version == version_before
