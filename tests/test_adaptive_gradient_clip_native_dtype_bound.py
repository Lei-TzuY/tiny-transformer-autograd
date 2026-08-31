import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def _l2_float64(array):
    values = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values)))


def test_float32_rounding_never_leaves_stored_gradient_above_clip_bound():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0], dtype=np.float32)

    with np.errstate(all="raise"):
        changed = adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert changed == 1
    assert parameter.grad.dtype == np.float32
    assert _l2_float64(parameter.grad) <= 0.5


def test_float64_path_keeps_the_existing_clipping_result():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0], dtype=np.float64)

    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 1
    np.testing.assert_allclose(parameter.grad, [0.3, 0.4], rtol=0.0, atol=1e-15)
