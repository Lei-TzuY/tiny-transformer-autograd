import numpy as np
import pytest

from engine.gradient_accumulator import GradientAccumulator
from engine.tensor import Tensor


class MutateThenFailTensor(Tensor):
    def __init__(self, data, requires_grad=True):
        object.__setattr__(self, "fail_next_grad_write", False)
        super().__init__(data, requires_grad=requires_grad)

    def __setattr__(self, name, value):
        if name == "grad" and getattr(self, "fail_next_grad_write", False):
            object.__setattr__(self, name, value)
            object.__setattr__(self, "fail_next_grad_write", False)
            raise RuntimeError("injected grad write failure")
        object.__setattr__(self, name, value)


def test_copy_to_grads_rolls_back_mutate_then_raise_assignment():
    first = Tensor([0.0], requires_grad=True)
    second = MutateThenFailTensor([0.0], requires_grad=True)
    first.grad[...] = [11.0]
    second.grad[...] = [22.0]
    first_original = first.grad
    second_original = second.grad

    accumulator = GradientAccumulator([first, second])
    first.grad[...] = [1.0]
    second.grad[...] = [2.0]
    accumulator.accumulate()

    # Restore caller-owned gradient objects before testing the commit path.
    first.grad = first_original
    second.grad = second_original
    first.grad[...] = [11.0]
    second.grad[...] = [22.0]
    second.fail_next_grad_write = True

    with pytest.raises(RuntimeError, match="injected grad write failure"):
        accumulator.copy_to_grads()

    assert first.grad is first_original
    assert second.grad is second_original
    np.testing.assert_array_equal(first.grad, [11.0])
    np.testing.assert_array_equal(second.grad, [22.0])
    np.testing.assert_array_equal(accumulator.average_gradients()[0], [1.0])
    np.testing.assert_array_equal(accumulator.average_gradients()[1], [2.0])
