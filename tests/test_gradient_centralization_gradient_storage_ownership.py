"""Gradient centralization must not write through another Tensor's managed storage."""

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def test_rejects_external_tensor_managed_gradient_storage_before_write():
    parameter = Tensor([[10.0, 20.0]], requires_grad=True)
    external = Tensor([[1.0, 3.0]], requires_grad=False)
    parameter.grad = external.data

    external_data = external.data
    external_values = np.array(external_data, copy=True)
    external_version = external._version

    with pytest.raises(ValueError, match="Tensor-managed storage"):
        centralize_gradients_([parameter])

    assert parameter.grad is external_data
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_allows_external_tensor_managed_gradient_storage_when_centralization_is_noop():
    parameter = Tensor([[2.0, 2.0]], requires_grad=True)
    external = Tensor([[4.0, 4.0]], requires_grad=False)
    parameter.grad = external.data

    external_data = external.data
    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert centralize_gradients_([parameter]) == 0

    assert parameter.grad is external_data
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version
