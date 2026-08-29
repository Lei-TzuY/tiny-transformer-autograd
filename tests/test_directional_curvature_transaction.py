import numpy as np
import pytest

from engine.directional_curvature import directional_curvature
from engine.tensor import Tensor


class MutateThenRaiseOnce(np.ndarray):
    def __new__(cls, values):
        obj = np.array(values, dtype=np.float64).view(cls)
        obj.calls = 0
        return obj

    def __array_finalize__(self, source):
        self.calls = getattr(source, "calls", 0)

    def __setitem__(self, key, value):
        self.calls += 1
        super().__setitem__(key, value)
        if self.calls == 1:
            raise RuntimeError("injected write failure")


class IgnoreFirstWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.array(values, dtype=np.float64).view(cls)
        obj.calls = 0
        return obj

    def __array_finalize__(self, source):
        self.calls = getattr(source, "calls", 0)

    def __setitem__(self, key, value):
        self.calls += 1
        if self.calls == 1:
            return
        super().__setitem__(key, value)


def test_late_mutate_then_raise_write_restores_all_parameter_values():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = MutateThenRaiseOnce([2.0])
    first_before = first.data.copy()
    second_before = np.array(second.data, copy=True)

    with pytest.raises(RuntimeError, match="injected write failure"):
        directional_curvature(
            lambda: first.data[0] ** 2 + second.data[0] ** 2,
            [first, second],
            [np.array([1.0]), np.array([1.0])],
            step=0.5,
        )

    np.testing.assert_array_equal(first.data, first_before)
    np.testing.assert_array_equal(second.data, second_before)


def test_silent_write_is_detected_and_restored():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = IgnoreFirstWrite([2.0])
    before = np.array(parameter.data, copy=True)

    with pytest.raises(RuntimeError, match="did not retain"):
        directional_curvature(
            lambda: parameter.data[0] ** 2,
            parameter,
            np.array([1.0]),
            step=0.5,
        )

    np.testing.assert_array_equal(parameter.data, before)


def test_callback_shape_replacement_is_detected_and_entry_shape_restored():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    original_storage = parameter.data
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls == 2:
            parameter.data = np.array([[99.0]])
        return float(np.sum(parameter.data))

    with pytest.raises(RuntimeError, match="replaced parameter 0 storage"):
        directional_curvature(
            loss,
            parameter,
            np.array([1.0, 0.0]),
            step=0.5,
        )

    assert parameter.data is original_storage
    assert parameter.shape == (2,)
    np.testing.assert_array_equal(parameter.data, [2.0, 3.0])


def test_callback_value_mutation_is_detected_even_if_storage_binding_is_preserved():
    parameter = Tensor([2.0], requires_grad=True)
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        if calls == 2:
            parameter.data[...] = [88.0]
        return parameter.data[0] ** 2

    with pytest.raises(RuntimeError, match="modified parameter 0"):
        directional_curvature(
            loss,
            parameter,
            np.array([1.0]),
            step=0.5,
        )
    np.testing.assert_array_equal(parameter.data, [2.0])


def test_failure_never_rewinds_tensor_versions():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = MutateThenRaiseOnce([2.0])
    first_version = first._version
    second_version = second._version

    with pytest.raises(RuntimeError):
        directional_curvature(
            lambda: 0.0,
            [first, second],
            [np.array([1.0]), np.array([1.0])],
            step=0.5,
        )

    assert first._version >= first_version
    assert second._version >= second_version
    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
