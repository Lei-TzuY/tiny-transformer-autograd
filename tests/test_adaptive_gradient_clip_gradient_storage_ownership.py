import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def test_rejects_tensor_managed_gradient_storage_before_clipping_write():
    parameter = Tensor(np.array([10.0]), requires_grad=True)
    external = Tensor(np.array([4.0]), requires_grad=True)
    parameter.grad = external.data

    external_values = external.data.copy()
    external_version = external._version

    with pytest.raises(ValueError, match="gradient.*Tensor-managed storage"):
        adaptive_clip_grad_([parameter], clip_factor=0.01, eps=1e-3)

    assert parameter.grad is external.data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version
