import numpy as np
import pytest

from engine import Tensor, centralize_gradients_


def _parameter_with_gradient():
    parameter = Tensor([[2.0, 4.0]], requires_grad=True)
    parameter.grad = np.array([[1.0, 3.0]], dtype=np.float64)
    return parameter


@pytest.mark.parametrize("bad_version", [True, np.int64(0), 1.5, "0"])
def test_centralization_rejects_noncanonical_mutation_version_before_write(bad_version):
    parameter = _parameter_with_gradient()
    gradient = parameter.grad
    before = gradient.copy()
    parameter._version = bad_version

    with pytest.raises(TypeError, match="parameter 0 mutation version must be a non-negative integer"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter._version is bad_version


def test_centralization_rejects_negative_mutation_version_before_write():
    parameter = _parameter_with_gradient()
    gradient = parameter.grad
    before = gradient.copy()
    parameter._version = -1

    with pytest.raises(ValueError, match="parameter 0 mutation version must be non-negative"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter._version == -1
