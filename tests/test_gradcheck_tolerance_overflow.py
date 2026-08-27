"""Regression coverage for oversized gradcheck tolerance values."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradcheck import gradcheck
from engine.tensor import Tensor


@pytest.mark.parametrize("name", ["eps", "atol", "rtol"])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_gradcheck_normalizes_tolerance_float_overflow(name, value):
    x = Tensor([1.0, 2.0])
    calls = 0

    def function(tensor):
        nonlocal calls
        calls += 1
        return tensor * tensor

    with pytest.raises(ValueError, match=rf"gradcheck {name} must be finite"):
        gradcheck(function, x, **{name: value})

    assert calls == 0


def test_gradcheck_tolerance_overflow_does_not_touch_parameter_state():
    parameter = Tensor([2.0, -3.0], requires_grad=True)
    parameter.grad = parameter.data.copy()
    grad_buffer = parameter.grad
    data_before = parameter.data.copy()
    grad_before = parameter.grad.copy()
    version_before = parameter._version

    with pytest.raises(ValueError, match="gradcheck atol must be finite"):
        gradcheck(
            lambda: parameter * parameter,
            parameters=[parameter],
            atol=10**400,
        )

    assert parameter._version == version_before
    assert parameter.grad is grad_buffer
    assert (parameter.data == data_before).all()
    assert (parameter.grad == grad_before).all()
