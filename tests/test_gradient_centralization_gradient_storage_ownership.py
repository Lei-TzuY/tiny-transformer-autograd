"""Gradient centralization must not write through another Tensor's managed storage."""

import weakref

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


class _GradientView(np.ndarray):
    """Ordinary ndarray subclass used to preserve a foreign storage base chain."""


class _WeakrefTarget:
    """Short-lived target used to construct a deliberately dead owner reference."""


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


def test_rejects_tensor_managed_gradient_with_dead_owner_reference_before_write():
    parameter = Tensor([[10.0, 20.0]], requires_grad=True)
    external = Tensor([[1.0, 3.0]], requires_grad=False)
    external_data = external.data
    parameter.grad = external_data

    target = _WeakrefTarget()
    dead_owner_ref = weakref.ref(target)
    del target
    assert dead_owner_ref() is None
    external_data._owner_ref = dead_owner_ref

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    with pytest.raises(ValueError, match="ownership metadata"):
        centralize_gradients_([parameter])

    assert parameter.grad is external_data
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_rejects_tensor_managed_gradient_with_nonweak_owner_metadata_before_write():
    parameter = Tensor([[10.0, 20.0]], requires_grad=True)
    external = Tensor([[1.0, 3.0]], requires_grad=False)
    external_data = external.data
    parameter.grad = external_data
    external_data._owner_ref = object()

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    with pytest.raises(TypeError, match="ownership metadata"):
        centralize_gradients_([parameter])

    assert parameter.grad is external_data
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_rejects_plain_ndarray_view_of_external_tensor_storage_before_write():
    parameter = Tensor([[10.0, 20.0]], requires_grad=True)
    external = Tensor([[1.0, 3.0]], requires_grad=False)
    external_data = external.data
    plain_view = external_data.view(np.ndarray)
    parameter.grad = plain_view

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert type(plain_view) is np.ndarray
    assert np.shares_memory(plain_view, external_data)

    with pytest.raises(ValueError, match="must own its storage"):
        centralize_gradients_([parameter])

    assert parameter.grad is plain_view
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_rejects_ndarray_subclass_view_of_external_tensor_storage_before_write():
    parameter = Tensor([[10.0, 20.0]], requires_grad=True)
    external = Tensor([[1.0, 3.0]], requires_grad=False)
    external_data = external.data
    subclass_view = external_data.view(_GradientView)
    parameter.grad = subclass_view

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert type(subclass_view) is _GradientView
    assert np.shares_memory(subclass_view, external_data)

    with pytest.raises(ValueError, match="Tensor-managed storage"):
        centralize_gradients_([parameter])

    assert parameter.grad is subclass_view
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_allows_external_tensor_managed_gradient_storage_when_centralization_is_noop():
    parameter = Tensor([[2.0, 2.0]], requires_grad=True)
    external = Tensor([[-1.0, 1.0]], requires_grad=False)
    parameter.grad = external.data

    external_data = external.data
    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert centralize_gradients_([parameter]) == 0

    assert parameter.grad is external_data
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_allows_plain_ndarray_view_of_external_tensor_storage_when_noop():
    parameter = Tensor([[2.0, 2.0]], requires_grad=True)
    external = Tensor([[-1.0, 1.0]], requires_grad=False)
    external_data = external.data
    plain_view = external_data.view(np.ndarray)
    parameter.grad = plain_view

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert centralize_gradients_([parameter]) == 0

    assert parameter.grad is plain_view
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version


def test_allows_ndarray_subclass_view_of_external_tensor_storage_when_noop():
    parameter = Tensor([[2.0, 2.0]], requires_grad=True)
    external = Tensor([[-1.0, 1.0]], requires_grad=False)
    external_data = external.data
    subclass_view = external_data.view(_GradientView)
    parameter.grad = subclass_view

    external_values = np.array(external_data, copy=True)
    external_version = external._version

    assert centralize_gradients_([parameter]) == 0

    assert parameter.grad is subclass_view
    assert external.data is external_data
    np.testing.assert_array_equal(external.data, external_values)
    assert external._version == external_version
