import numpy as np
import pytest

from engine.pcgrad import PCGradBuffer
from engine.tensor import Tensor


class _FailOnceGradTensor(Tensor):
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_fail_next_grad_write", False)
        super().__init__(*args, **kwargs)

    def fail_next_grad_write(self):
        object.__setattr__(self, "_fail_next_grad_write", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "grad" and getattr(self, "_fail_next_grad_write", False):
            object.__setattr__(self, "_fail_next_grad_write", False)
            raise RuntimeError("injected grad assignment failure")


class _AlwaysFailGradTensor(Tensor):
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_fail_grad_writes", False)
        super().__init__(*args, **kwargs)

    def start_failing_grad_writes(self):
        object.__setattr__(self, "_fail_grad_writes", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "grad" and getattr(self, "_fail_grad_writes", False):
            raise RuntimeError("persistent grad assignment failure")


def test_copy_to_grads_rolls_back_prior_and_failing_grad_references():
    first = Tensor([0.0], requires_grad=True)
    second = _FailOnceGradTensor([0.0], requires_grad=True)
    first_original = np.asarray([7.0])
    second_original = np.asarray([8.0])
    first.grad = first_original
    second.grad = second_original

    pcgrad = PCGradBuffer([first, second])
    first.grad = np.asarray([1.0])
    second.grad = np.asarray([2.0])
    pcgrad.capture()
    first.grad = first_original
    second.grad = second_original
    before_tasks = pcgrad.task_gradients()

    second.fail_next_grad_write()
    with pytest.raises(RuntimeError, match="injected grad assignment failure"):
        pcgrad.copy_to_grads()

    assert first.grad is first_original
    assert second.grad is second_original
    np.testing.assert_array_equal(first.grad, [7.0])
    np.testing.assert_array_equal(second.grad, [8.0])
    after_tasks = pcgrad.task_gradients()
    np.testing.assert_array_equal(after_tasks[0][0], before_tasks[0][0])
    np.testing.assert_array_equal(after_tasks[0][1], before_tasks[0][1])


def test_copy_to_grads_reports_rollback_failure_explicitly():
    first = Tensor([0.0], requires_grad=True)
    second = _AlwaysFailGradTensor([0.0], requires_grad=True)
    pcgrad = PCGradBuffer([first, second])
    first.grad = np.asarray([1.0])
    second.grad = np.asarray([2.0])
    pcgrad.capture()

    first.grad = np.asarray([10.0])
    second.grad = np.asarray([20.0])
    second.start_failing_grad_writes()

    with pytest.raises(RuntimeError, match="PCGrad gradient rollback failed"):
        pcgrad.copy_to_grads()
