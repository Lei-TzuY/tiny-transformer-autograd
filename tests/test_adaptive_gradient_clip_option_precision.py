import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class DispatchTrackingFloat(np.float64):
    ufunc_calls = 0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        type(self).ufunc_calls += 1
        raise RuntimeError("unexpected NumPy scalar subclass dispatch")


def test_wider_finite_options_outside_float64_are_range_errors():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    too_large = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    assert np.isfinite(too_large)

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="clip_factor must fit float64"):
            adaptive_clip_grad_(parameter, clip_factor=too_large)
        with pytest.raises(ValueError, match="eps must fit float64"):
            adaptive_clip_grad_(parameter, eps=too_large)


def test_representable_wider_options_remain_supported():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([0.0])

    with np.errstate(all="raise"):
        assert (
            adaptive_clip_grad_(
                parameter,
                clip_factor=np.longdouble("0.125"),
                eps=np.longdouble("0.001"),
            )
            == 0
        )


def test_numpy_float_subclass_options_do_not_dispatch_ufunc_hooks():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    DispatchTrackingFloat.ufunc_calls = 0

    with np.errstate(all="raise"):
        assert adaptive_clip_grad_(parameter, clip_factor=DispatchTrackingFloat(0.125)) == 0
        assert adaptive_clip_grad_(parameter, eps=DispatchTrackingFloat(0.001)) == 0

    assert DispatchTrackingFloat.ufunc_calls == 0
