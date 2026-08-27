"""Regression tests for strict gradcheck tolerance types."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradcheck import gradcheck
from engine.tensor import Tensor


@pytest.mark.parametrize("name", ["eps", "atol", "rtol"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_gradcheck_rejects_boolean_tolerances(name, value):
    kwargs = {name: value}

    with pytest.raises(TypeError, match=rf"gradcheck {name} must be a real number"):
        gradcheck(lambda x: x * x, Tensor([0.5]), **kwargs)


def test_boolean_atol_cannot_hide_an_incorrect_backward_rule():
    def broken_identity(x):
        out = Tensor(
            x.data.copy(),
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="broken_identity",
        )

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                x.grad += 0.0 * out.grad

        out._backward = _backward
        return out

    # Before this fix, atol=True was interpreted as 1.0 and could let this
    # unit-sized derivative error pass. Boolean tolerance values are now
    # rejected before the checker performs any analytical/numerical work.
    with pytest.raises(TypeError, match="gradcheck atol must be a real number"):
        gradcheck(broken_identity, Tensor([0.5]), atol=True, rtol=0.0)
