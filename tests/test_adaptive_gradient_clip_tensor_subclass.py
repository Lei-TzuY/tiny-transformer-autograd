import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class HostileTensor(Tensor):
    """Tensor subclass whose data getter must not run during AGC validation."""

    data_reads = 0

    @property
    def data(self):
        type(self).data_reads += 1
        return self._data

    @data.setter
    def data(self, value):
        Tensor.data.fset(self, value)


def test_tensor_subclass_is_rejected_before_overridden_data_getter_runs():
    parameter = HostileTensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0], dtype=np.float64)
    parameter.grad = gradient
    HostileTensor.data_reads = 0

    with pytest.raises(TypeError, match="parameter 0 must be a Tensor"):
        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert HostileTensor.data_reads == 0
    np.testing.assert_array_equal(gradient, np.array([6.0, 8.0]))


def test_builtin_tensor_keeps_existing_clipping_semantics():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0], dtype=np.float64)
    parameter.grad = gradient

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert changed == 1
    np.testing.assert_allclose(gradient, np.array([0.3, 0.4]), rtol=0.0, atol=1e-15)
