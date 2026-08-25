"""Regression tests for the Tensor requires_grad boolean contract."""

import numpy as np
import pytest

from engine.grad_mode import no_grad
from engine.tensor import Tensor


class _ForbiddenArrayConversion:
    def __array__(self, dtype=None, copy=None):
        pytest.fail("invalid requires_grad reached Tensor data conversion")


@pytest.mark.parametrize(
    "requires_grad",
    [0, 1, 1.0, "yes", None, [], np.array(True)],
)
def test_tensor_rejects_non_boolean_requires_grad_before_data_conversion(
    requires_grad,
):
    with pytest.raises(TypeError, match="requires_grad"):
        Tensor(_ForbiddenArrayConversion(), requires_grad=requires_grad)


@pytest.mark.parametrize(
    "requires_grad,expected",
    [
        (False, False),
        (True, True),
        (np.bool_(False), False),
        (np.bool_(True), True),
    ],
)
def test_tensor_accepts_and_canonicalizes_boolean_requires_grad(
    requires_grad, expected
):
    tensor = Tensor([1.0, -2.0], requires_grad=requires_grad)

    assert type(tensor.requires_grad) is bool
    assert tensor.requires_grad is expected
    if expected:
        np.testing.assert_array_equal(tensor.grad, np.zeros(2))
    else:
        assert tensor.grad is None


def test_numpy_boolean_trainable_leaf_remains_trainable_inside_no_grad():
    with no_grad():
        tensor = Tensor([3.0], requires_grad=np.bool_(True))

    assert tensor.requires_grad is True
    np.testing.assert_array_equal(tensor.grad, np.zeros(1))
