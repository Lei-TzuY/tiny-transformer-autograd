import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutateThenRaise(np.ndarray):
    def __new__(cls, values, failures=1):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.failures = failures
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.failures = getattr(source, "failures", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("injected gradient write failure")


class RejectWrites(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float64).view(cls)

    def __setitem__(self, key, value):
        raise RuntimeError("gradient destination rejected write")


class DropFirstWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.drop_count = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.drop_count = getattr(source, "drop_count", 0)

    def __setitem__(self, key, value):
        if self.drop_count > 0:
            self.drop_count -= 1
            return
        super().__setitem__(key, value)


def test_late_mutate_then_raise_rolls_back_every_attempted_gradient():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = MutateThenRaise([6.0, 8.0], failures=1)
    first_ref = first.grad
    second_ref = second.grad
    first_before = first.grad.copy()
    second_before = second.grad.copy()
    first_version = first._version
    second_version = second._version

    with pytest.raises(RuntimeError, match="injected gradient write failure"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    assert first.grad is first_ref
    assert second.grad is second_ref
    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)
    assert first._version == first_version
    assert second._version == second_version


def test_rejected_unmodified_commit_re_raises_original_failure_without_rollback_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = RejectWrites([6.0, 8.0])
    gradient_ref = parameter.grad
    before = parameter.grad.copy()

    with pytest.raises(RuntimeError, match="gradient destination rejected write"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, before)


def test_silent_write_drop_is_detected_and_rolled_back():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = DropFirstWrite([6.0, 8.0])
    gradient_ref = parameter.grad
    before = parameter.grad.copy()

    with pytest.raises(RuntimeError, match="write failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, before)


def test_rollback_failure_is_explicit():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = MutateThenRaise([6.0, 8.0], failures=2)

    with pytest.raises(RuntimeError, match="adaptive gradient clipping rollback failed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)


def test_float32_candidate_underflow_is_warning_free_and_preserves_dtype():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([1.0], dtype=np.float32)
    gradient_ref = parameter.grad

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-50)

    assert changed == 1
    assert parameter.grad is gradient_ref
    assert parameter.grad.dtype == np.float32
    assert parameter.grad[0] == 0.0


def test_rejected_transaction_preserves_numpy_rng_and_parameter_data():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    second.grad = MutateThenRaise([6.0, 8.0], failures=1)
    first_data = first.data.copy()
    second_data = second.data.copy()
    rng_before = np.random.get_state()

    with pytest.raises(RuntimeError):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    rng_after = np.random.get_state()
    np.testing.assert_array_equal(first.data, first_data)
    np.testing.assert_array_equal(second.data, second_data)
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
