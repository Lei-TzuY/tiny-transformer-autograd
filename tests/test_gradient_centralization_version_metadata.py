import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def _parameter_with_gradient():
    parameter = Tensor([[2.0, 4.0]], requires_grad=True)
    parameter.grad = np.array([[1.0, 3.0]], dtype=np.float64)
    return parameter


class _VersionCorruptingGradient(np.ndarray):
    def __new__(cls, values, parameter):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj._parameter = parameter
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._parameter = getattr(obj, "_parameter", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self._parameter is not None:
            self._parameter._version = np.int64(self._parameter._version)


@pytest.mark.parametrize("bad_version", [True, np.int64(0), 1.5, "0"])
def test_centralization_rejects_noncanonical_mutation_version_before_write(bad_version):
    parameter = _parameter_with_gradient()
    gradient = parameter.grad
    before = gradient.copy()
    parameter._version = bad_version

    with pytest.raises(TypeError, match="parameter 0 mutation version must be a non-negative integer"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter._version is bad_version


def test_centralization_rejects_negative_mutation_version_before_write():
    parameter = _parameter_with_gradient()
    gradient = parameter.grad
    before = gradient.copy()
    parameter._version = -1

    with pytest.raises(ValueError, match="parameter 0 mutation version must be non-negative"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter._version == -1


def test_centralization_rejects_mutation_version_type_drift_during_write():
    parameter = _parameter_with_gradient()
    entry_version = parameter._version
    gradient = _VersionCorruptingGradient(parameter.grad, parameter)
    before = np.array(gradient, copy=True)
    parameter.grad = gradient

    with pytest.raises(RuntimeError, match="mutation version changed for parameter 0"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert type(parameter._version) is int
    assert parameter._version == entry_version
