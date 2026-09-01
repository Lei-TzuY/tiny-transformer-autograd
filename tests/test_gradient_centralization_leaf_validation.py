"""Public gradient centralization accepts model leaves, not graph intermediates."""

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def test_centralization_rejects_nonleaf_before_touching_gradient():
    leaf = Tensor([[1.0, 2.0]], requires_grad=True)
    intermediate = leaf * 2.0
    gradient = intermediate.grad
    gradient[...] = np.array([[3.0, 1.0]])
    before = gradient.copy()

    with pytest.raises(ValueError, match=r"parameter 0 must be a leaf Tensor"):
        centralize_gradients_(intermediate)

    assert intermediate.grad is gradient
    np.testing.assert_array_equal(gradient, before)


def test_centralization_accepts_explicit_leaf_parameter():
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    gradient = parameter.grad
    gradient[...] = np.array([[3.0, 1.0]])

    changed = centralize_gradients_(parameter)

    assert changed == 1
    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, np.array([[1.0, -1.0]]))
