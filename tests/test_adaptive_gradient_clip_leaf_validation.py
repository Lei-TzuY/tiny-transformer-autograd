import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def test_adaptive_clip_rejects_non_leaf_tensor_before_gradient_write():
    leaf = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    result = leaf * 2.0
    result.grad[...] = np.array([6.0, 8.0])

    result_gradient = result.grad.copy()
    leaf_gradient = leaf.grad.copy()

    with pytest.raises(ValueError, match="parameter 0 must be a leaf Tensor"):
        adaptive_clip_grad_(result, clip_factor=0.1, eps=1e-3)

    np.testing.assert_array_equal(result.grad, result_gradient)
    np.testing.assert_array_equal(leaf.grad, leaf_gradient)


def test_adaptive_clip_still_accepts_explicit_leaf_tensor():
    parameter = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    parameter.grad[...] = np.array([6.0, 8.0])

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert changed == 1
    np.testing.assert_allclose(parameter.grad, np.array([0.3, 0.4]), rtol=0.0, atol=1e-15)
