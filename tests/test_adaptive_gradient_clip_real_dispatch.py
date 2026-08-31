from fractions import Fraction

import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class FractionSubclass(Fraction):
    float_calls = 0

    def __float__(self):
        type(self).float_calls += 1
        raise RuntimeError("unexpected Fraction subclass dispatch")


def _idle_parameter():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([0.0])
    return parameter


def test_exact_fraction_options_remain_supported():
    parameter = _idle_parameter()

    assert adaptive_clip_grad_(parameter, clip_factor=Fraction(1, 8)) == 0
    assert adaptive_clip_grad_(parameter, eps=Fraction(1, 1000)) == 0


def test_fraction_subclass_options_fail_closed_without_float_dispatch():
    parameter = _idle_parameter()
    FractionSubclass.float_calls = 0

    with pytest.raises(TypeError, match="clip_factor"):
        adaptive_clip_grad_(parameter, clip_factor=FractionSubclass(1, 8))
    with pytest.raises(TypeError, match="eps"):
        adaptive_clip_grad_(parameter, eps=FractionSubclass(1, 1000))

    assert FractionSubclass.float_calls == 0
