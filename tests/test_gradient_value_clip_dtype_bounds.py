import numpy as np
import pytest

from engine.gradient_value_clip import clip_grad_value_
from engine.tensor import Tensor


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_huge_finite_limit_is_warning_free_noop_for_narrow_gradients(dtype):
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    maximum = np.finfo(dtype).max
    parameter.grad = np.array([maximum, -maximum], dtype=dtype)
    gradient = parameter.grad
    before = gradient.copy()

    with np.errstate(all="raise"):
        assert clip_grad_value_(parameter, 1e300) == 0

    assert parameter.grad is gradient
    assert parameter.grad.dtype == dtype
    np.testing.assert_array_equal(parameter.grad, before)


def test_subnormal_limit_narrowing_is_warning_free_and_preserves_dtype():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([1.0, -1.0], dtype=np.float32)
    gradient = parameter.grad
    limit = np.finfo(np.float64).smallest_subnormal

    with np.errstate(all="raise"):
        assert clip_grad_value_(parameter, limit) == 1

    assert parameter.grad is gradient
    assert parameter.grad.dtype == np.float32
    np.testing.assert_array_equal(parameter.grad, np.array([0.0, -0.0], dtype=np.float32))
