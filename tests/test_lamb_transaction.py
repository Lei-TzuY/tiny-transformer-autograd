import numpy as np
import pytest

from engine.lamb import LAMB
from engine.tensor import Tensor


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
    return LAMB(
        parameters,
        lr=0.05,
        betas=(0.5, 0.5),
        eps=1e-6,
        weight_decay=0.0,
    )


def test_late_mutate_then_raise_rolls_back_earlier_parameter_and_state():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    second._data = MutateThenRaise([3.0])
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    optimizer = _optimizer([first, second])
    first_version = first._version

    with pytest.raises(RuntimeError, match="injected write failure"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, [2.0])
    np.testing.assert_array_equal(second.data, [3.0])
    assert optimizer.steps == (0, 0)
    assert first._version >= first_version + 2


def test_failed_transaction_does_not_commit_candidate_moments():
    first = Tensor([2.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 5.0], requires_grad=True)
    second._data = MutateThenRaise([3.0, 5.0])
    first.grad = np.array([1.0, 2.0])
    second.grad = np.array([2.0, 1.0])
    optimizer = _optimizer([first, second])
    before = optimizer.state_dict()

    with pytest.raises(RuntimeError):
        optimizer.step()

    after = optimizer.state_dict()
    for index in (0, 1):
        assert after["states"][index]["step"] == before["states"][index]["step"] == 0
        np.testing.assert_array_equal(after["states"][index]["m"], before["states"][index]["m"])
        np.testing.assert_array_equal(after["states"][index]["v"], before["states"][index]["v"])
        assert after["states"][index]["v_scale"] == before["states"][index]["v_scale"]


def test_silent_ignored_write_is_detected_and_rolled_back():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = IgnoreFirstWrite([2.0])
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)

    with pytest.raises(RuntimeError, match="rejected LAMB update"):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert optimizer.steps == (0,)


def test_rollback_postcondition_failure_is_explicit():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = CorruptEveryWrite([2.0])
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)

    with pytest.raises(RuntimeError, match="LAMB parameter rollback failed"):
        optimizer.step()

    assert optimizer.steps == (0,)


def test_failure_does_not_modify_live_gradients_or_rng():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    second._data = MutateThenRaise([3.0])
    first_grad = np.array([1.0])
    second_grad = np.array([2.0])
    first.grad = first_grad
    second.grad = second_grad
    optimizer = _optimizer([first, second])
    rng_before = np.random.get_state()

    with pytest.raises(RuntimeError):
        optimizer.step()

    assert first.grad is first_grad
    assert second.grad is second_grad
    np.testing.assert_array_equal(first.grad, [1.0])
    np.testing.assert_array_equal(second.grad, [2.0])
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_candidate_overflow_on_late_parameter_fails_before_first_write():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([np.finfo(np.float64).max], requires_grad=True)
    first.grad = np.array([1.0])
    # This direction grows +max because parameter and update directions oppose.
    second.grad = np.array([-1.0])
    optimizer = LAMB(
        [first, second],
        lr=1.0,
        betas=(0.0, 0.0),
        eps=1.0,
        weight_decay=0.0,
    )
    first_before = first.data.copy()

    with pytest.raises(ValueError, match="representable"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, first_before)
    assert optimizer.steps == (0, 0)


def test_zero_grad_preserves_gradient_array_identity():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    gradient = np.array([4.0, 5.0])
    parameter.grad = gradient
    optimizer = LAMB(parameter)

    optimizer.zero_grad()

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [0.0, 0.0])
