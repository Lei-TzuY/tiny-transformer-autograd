import numpy as np
import pytest

from engine.adafactor import Adafactor
from engine.tensor import Tensor


_TINY = np.nextafter(0.0, 1.0)


class MutateThenRaise(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
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
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.calls = 0
        return obj

    def __array_finalize__(self, source):
        self.calls = getattr(source, "calls", 0)

    def __setitem__(self, key, value):
        self.calls += 1
        if self.calls == 1:
            return
        super().__setitem__(key, value)


class CorruptEveryWrite(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    def __array_finalize__(self, source):
        pass

    def __setitem__(self, key, value):
        incoming = np.asarray(value, dtype=np.float64)
        super().__setitem__(key, incoming + 1.0)


def _optimizer(parameters):
    return Adafactor(
        parameters,
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )


def test_late_mutate_then_raise_rolls_back_earlier_parameter_and_keeps_state_uncommitted():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = MutateThenRaise([2.0])
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = _optimizer([first, second])
    first_version = first._version

    with pytest.raises(RuntimeError, match="injected write failure"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [1.0])
    np.testing.assert_array_equal(second.data, [2.0])
    assert optimizer.steps == (0, 0)
    assert first._version > first_version


def test_silent_ignored_write_is_detected_and_original_value_remains():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = IgnoreFirstWrite([2.0])
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)

    with pytest.raises(RuntimeError, match="rejected Adafactor update"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert optimizer.steps == (0,)


def test_rollback_postcondition_failure_is_explicit():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = CorruptEveryWrite([2.0])
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)

    with pytest.raises(RuntimeError, match="Adafactor parameter rollback failed"):
        optimizer.step()

    assert optimizer.steps == (0,)


def test_failure_on_second_parameter_does_not_commit_first_moment_state():
    first = Tensor([5.0, 6.0], requires_grad=True)
    second = Tensor([7.0, 8.0], requires_grad=True)
    second._data = MutateThenRaise([7.0, 8.0])
    first.grad = np.array([2.0, 3.0])
    second.grad = np.array([4.0, 5.0])
    optimizer = Adafactor(
        [first, second],
        lr=0.01,
        beta2=0.5,
        eps=1e-20,
        clip_threshold=1.0,
    )
    before = optimizer.state_dict()

    with pytest.raises(RuntimeError, match="injected write failure"):
        optimizer.step()

    after = optimizer.state_dict()
    assert after["states"][0]["step"] == before["states"][0]["step"] == 0
    assert after["states"][1]["step"] == before["states"][1]["step"] == 0
    np.testing.assert_array_equal(after["states"][0]["v"], before["states"][0]["v"])
    np.testing.assert_array_equal(after["states"][1]["v"], before["states"][1]["v"])


def test_noop_parameter_candidate_does_not_require_writable_storage():
    parameter = Tensor([3.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    optimizer = Adafactor(parameter, lr=0.1, beta2=0.0, eps=1e-12)
    parameter.data.flags.writeable = False

    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [3.0])
    assert optimizer.steps == (1,)


def test_parameter_versions_are_not_rewound_after_failed_transaction():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = MutateThenRaise([2.0])
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = _optimizer([first, second])
    before = first._version

    with pytest.raises(RuntimeError):
        optimizer.step()

    assert first._version >= before + 2
    np.testing.assert_array_equal(first.data, [1.0])
