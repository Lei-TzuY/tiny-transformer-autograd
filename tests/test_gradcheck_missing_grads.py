"""Regressions for gradcheck parameters whose gradient buffer is absent."""

import numpy as np
import pytest

from engine.gradcheck import gradcheck
from engine.tensor import Tensor


def test_unused_parameter_with_none_grad_is_zero_and_restored():
    x = Tensor(np.array([0.3, -0.8], dtype=np.float64))
    parameter = Tensor(np.array([1.2, -0.4], dtype=np.float64), requires_grad=True)
    parameter.grad = None
    original_data = parameter.data.copy()

    assert gradcheck(
        lambda value: value * value,
        x,
        parameters=[("unused", parameter)],
    )

    assert parameter.grad is None
    np.testing.assert_array_equal(parameter.data, original_data)


def test_none_parameter_grad_is_restored_when_gradcheck_raises():
    x = Tensor(np.array([0.25], dtype=np.float64))
    parameter = Tensor(np.array([0.75], dtype=np.float64), requires_grad=True)
    parameter.grad = None

    with pytest.raises(TypeError, match="must return a Tensor"):
        gradcheck(
            lambda value: 1.0,
            x,
            parameters=[parameter],
        )

    assert parameter.grad is None
