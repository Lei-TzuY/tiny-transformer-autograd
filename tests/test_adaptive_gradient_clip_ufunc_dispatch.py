import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutateOnIsFinite(np.ndarray):
    """Expose validation-time ndarray ufunc dispatch as an observable side effect."""

    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.isfinite_calls = 0
        return obj

    def __array_finalize__(self, source):
        self.isfinite_calls = getattr(source, "isfinite_calls", 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.isfinite and method == "__call__":
            self.isfinite_calls += 1
            np.asarray(self)[...] = 0.5
        base_inputs = tuple(
            np.asarray(value) if isinstance(value, MutateOnIsFinite) else value
            for value in inputs
        )
        return getattr(ufunc, method)(*base_inputs, **kwargs)


def test_validation_does_not_dispatch_ndarray_subclass_ufuncs():
    parameter = Tensor(np.array([100.0]), requires_grad=True)
    gradient = MutateOnIsFinite([0.25])
    parameter.grad = gradient

    changed = adaptive_clip_grad_(parameter, clip_factor=0.01)

    assert changed == 0
    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, np.array([0.25]))
    assert gradient.isfinite_calls == 0
