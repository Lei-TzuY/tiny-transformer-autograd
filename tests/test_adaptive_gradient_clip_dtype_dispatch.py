import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutateOnDtype(np.ndarray):
    """Expose validation-time ndarray attribute dispatch as a side effect."""

    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.dtype_calls = 0
        return obj

    def __array_finalize__(self, source):
        self.dtype_calls = getattr(source, "dtype_calls", 0)

    @property
    def dtype(self):
        self.dtype_calls += 1
        np.asarray(self)[...] = 0.5
        return np.asarray(self).dtype


def test_candidate_dtype_does_not_dispatch_ndarray_subclass_property():
    parameter = Tensor(np.array([100.0]), requires_grad=True)
    gradient = MutateOnDtype([0.25])
    parameter.grad = gradient

    changed = adaptive_clip_grad_(parameter, clip_factor=0.01)

    assert changed == 0
    assert parameter.grad is gradient
    assert gradient.dtype_calls == 0
    np.testing.assert_array_equal(np.asarray(gradient), np.array([0.25]))
