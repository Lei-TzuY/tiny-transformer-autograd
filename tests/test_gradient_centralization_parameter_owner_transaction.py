"""Parameter-storage ownership regressions for gradient centralization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import centralize_gradients_
from engine.tensor import Tensor, _VersionedArray


class _CorruptParameterOwnerOnWrite(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._target.data._owner_ref = None


class _ReplaceParameterOwnerWithObjectOnWrite(np.ndarray):
    def __new__(cls, values, target):
        array = np.asarray(values, dtype=np.float64).view(cls)
        array._target = target
        return array

    def __array_finalize__(self, obj):
        self._target = getattr(obj, "_target", None)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._target.data._owner_ref = object()


class _DerivedVersionedArray(_VersionedArray):
    pass


def test_gradient_write_cannot_detach_parameter_storage_owner():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _CorruptParameterOwnerOnWrite([[1.0, 3.0]], parameter)
    parameter.grad = gradient
    gradient_before = np.array(gradient, copy=True)
    owner_ref_before = parameter.data._owner_ref
    version_before = parameter._version

    with pytest.raises(RuntimeError, match="parameter data ownership changed"):
        centralize_gradients_([parameter])

    assert parameter.data._owner_ref is owner_ref_before
    assert owner_ref_before() is parameter
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)

    parameter.data[...] = 1.0
    assert parameter._version == version_before + 1


def test_gradient_write_cannot_replace_owner_with_non_weakref_metadata():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    gradient = _ReplaceParameterOwnerWithObjectOnWrite([[1.0, 3.0]], parameter)
    parameter.grad = gradient
    gradient_before = np.array(gradient, copy=True)
    owner_ref_before = parameter.data._owner_ref

    with pytest.raises(RuntimeError, match="parameter data ownership changed"):
        centralize_gradients_([parameter])

    assert parameter.data._owner_ref is owner_ref_before
    assert owner_ref_before() is parameter
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, gradient_before)


def test_parameter_data_requires_exact_tensor_managed_storage_type():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    parameter.grad[...] = [[1.0, 3.0]]
    gradient_before = np.array(parameter.grad, copy=True)
    derived = parameter.data.view(_DerivedVersionedArray)
    assert derived._owner_ref() is parameter
    parameter._data = derived

    with pytest.raises(TypeError, match="data must use Tensor-managed storage"):
        centralize_gradients_([parameter])

    np.testing.assert_array_equal(parameter.grad, gradient_before)


def test_parameter_storage_owner_must_be_a_real_weak_reference():
    parameter = Tensor(np.zeros((1, 2), dtype=np.float64), requires_grad=True)
    parameter.grad[...] = [[1.0, 3.0]]
    gradient_before = np.array(parameter.grad, copy=True)

    # A callable strong reference preserves writes today but violates the Tensor
    # storage ownership contract: _VersionedArray deliberately keeps only a weak
    # reference so array storage does not extend its owning Tensor's lifetime.
    parameter.data._owner_ref = lambda: parameter

    with pytest.raises(TypeError, match="ownership metadata must be a weak reference"):
        centralize_gradients_([parameter])

    np.testing.assert_array_equal(parameter.grad, gradient_before)
