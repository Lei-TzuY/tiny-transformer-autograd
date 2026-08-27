"""Public-boundary validation regressions for gradcheck."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradcheck import gradcheck
from engine.tensor import Tensor


def test_non_callable_target_wins_before_malformed_parameters():
    with pytest.raises(TypeError, match="gradcheck function must be callable"):
        gradcheck(7, parameters=object())


def test_non_callable_target_does_not_consume_parameter_iterator():
    parameter = Tensor([1.0], requires_grad=True)
    consumed = []

    def parameters():
        consumed.append(True)
        yield parameter

    iterator = parameters()
    with pytest.raises(TypeError, match="gradcheck function must be callable"):
        gradcheck(None, parameters=iterator)

    assert consumed == []
    assert next(iterator) is parameter


def test_callable_target_keeps_existing_parameter_validation():
    with pytest.raises(TypeError, match="gradcheck parameters must be an iterable"):
        gradcheck(lambda: Tensor(1.0), parameters=object())
