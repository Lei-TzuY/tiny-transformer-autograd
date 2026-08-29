import numpy as np
import pytest

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


class _MutateThenRaiseOnce(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.fail_next = True
        return obj

    def __array_finalize__(self, source):
        self.fail_next = getattr(source, "fail_next", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected write failure")


class _AlwaysMutateThenRaise(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        raise RuntimeError("persistent injected write failure")


class _IgnoreOnce(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.ignore_next = True
        return obj

    def __array_finalize__(self, source):
        self.ignore_next = getattr(source, "ignore_next", False)

    def __setitem__(self, key, value):
        if self.ignore_next:
            self.ignore_next = False
            return
        super().__setitem__(key, value)


def test_late_mutate_then_raise_rolls_back_all_attempted_parameters():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = _MutateThenRaiseOnce([2.0])
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )
    first_version = first._version

    with pytest.raises(RuntimeError, match="injected write failure"):
        snapshot.restore()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
    assert first._version >= first_version + 2


def test_silent_commit_rejection_is_detected_and_rolled_back():
    p = Tensor([1.0], requires_grad=True)
    p._data = _IgnoreOnce([1.0])
    snapshot = ParameterSnapshot(p, values=np.array([9.0]))

    with pytest.raises(RuntimeError, match="rejected snapshot values"):
        snapshot.restore()

    np.testing.assert_array_equal(p.data, [1.0])


def test_rollback_failure_is_explicit_and_earlier_parameter_is_still_restored():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = _AlwaysMutateThenRaise([2.0])
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )

    with pytest.raises(RuntimeError, match="parameter snapshot rollback failed"):
        snapshot.restore()

    np.testing.assert_array_equal(first.data, [1.0])


def test_installed_entry_failure_rolls_back_before_body_runs():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = _MutateThenRaiseOnce([2.0])
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )
    entered = False

    with pytest.raises(RuntimeError, match="injected write failure"):
        with snapshot.installed():
            entered = True

    assert entered is False
    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])


def test_installed_exit_restoration_is_best_effort_across_parameters():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )

    with pytest.raises(RuntimeError, match="parameter snapshot restoration failed"):
        with snapshot.installed():
            np.testing.assert_array_equal(first.data, [10.0])
            np.testing.assert_array_equal(second.data, [20.0])
            first._data = _AlwaysMutateThenRaise([10.0])
            second.data[...] = 99.0

    np.testing.assert_array_equal(second.data, [2.0])


def test_installed_exit_rebuilds_storage_when_body_introduces_overlap():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    snapshot = ParameterSnapshot(
        [first, second],
        values=[np.array([10.0, 20.0]), np.array([30.0, 40.0])],
    )

    with snapshot.installed():
        backing = np.array([100.0, 200.0, 300.0])
        first._data = backing[:2]
        second._data = backing[1:]
        assert np.shares_memory(first.data, second.data)

    np.testing.assert_array_equal(first.data, [1.0, 2.0])
    np.testing.assert_array_equal(second.data, [3.0, 4.0])
    assert not np.shares_memory(first.data, second.data)


def test_body_exception_survives_when_restoration_succeeds():
    p = Tensor([1.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([2.0]))

    with pytest.raises(ValueError, match="original body error"):
        with snapshot.installed():
            raise ValueError("original body error")

    np.testing.assert_array_equal(p.data, [1.0])


def test_restore_rollback_never_rewinds_tensor_version_history():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = _MutateThenRaiseOnce([2.0])
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )
    before = first._version

    with pytest.raises(RuntimeError):
        snapshot.restore()

    assert first._version > before
    np.testing.assert_array_equal(first.data, [1.0])
