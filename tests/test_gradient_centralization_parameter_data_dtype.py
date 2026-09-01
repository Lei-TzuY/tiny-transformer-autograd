import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def test_centralization_rejects_non_float64_parameter_storage_before_gradient_write():
    parameter = Tensor([[2.0, 4.0]], requires_grad=True)
    gradient = np.array([[1.0, 3.0]], dtype=np.float64)
    parameter.grad = gradient
    before = gradient.copy()

    np.ndarray.dtype.__set__(parameter.data, np.int64)
    assert parameter.data.dtype == np.dtype(np.int64)
    assert parameter.data.shape == (1, 2)

    with pytest.raises(TypeError, match="parameter 0 data must have dtype float64"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter.data.dtype == np.dtype(np.int64)
