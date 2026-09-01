import warnings

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def test_centralization_rejects_non_float64_parameter_storage_before_gradient_write():
    parameter = Tensor([[2.0, 4.0]], requires_grad=True)
    gradient = np.array([[1.0, 3.0]], dtype=np.float64)
    parameter.grad = gradient
    before = gradient.copy()

    # Deliberately manufacture an impossible-through-public-API Tensor state.
    # NumPy 2.5 deprecates direct dtype mutation, so suppress only the fixture's
    # warning while keeping the repository's warnings-as-errors policy intact.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.dtype.__set__(parameter.data, np.int64)
    assert parameter.data.dtype == np.dtype(np.int64)
    assert parameter.data.shape == (1, 2)

    with pytest.raises(TypeError, match="parameter 0 data must have dtype float64"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter.data.dtype == np.dtype(np.int64)
