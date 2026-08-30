import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class LyingArrayEqual(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.array_equal_calls = 0
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.array_equal_calls = getattr(source, "array_equal_calls", 0)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.array_equal:
            self.array_equal_calls += 1
            return True
        return super().__array_function__(func, types, args, kwargs)


def test_gradient_subclass_cannot_lie_about_candidate_equality():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = LyingArrayEqual([6.0, 8.0])
    gradient = parameter.grad

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert changed == 1
    assert parameter.grad is gradient
    np.testing.assert_array_equal(np.asarray(gradient), np.array([0.3, 0.4]))
    assert gradient.array_equal_calls == 0
