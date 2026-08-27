"""Public-boundary validation regressions for gradient recomputation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.grad_mode import no_grad
from engine.recompute import recompute
from engine.tensor import Tensor


def _input():
    return Tensor([1.0, -2.0], requires_grad=True)


@pytest.mark.parametrize("disabled", [False, True])
def test_recompute_rejects_non_callable_function_explicitly(disabled):
    x = _input()

    if disabled:
        with no_grad(), pytest.raises(TypeError, match="function must be callable"):
            recompute(7, x)
    else:
        with pytest.raises(TypeError, match="function must be callable"):
            recompute(7, x)


@pytest.mark.parametrize(
    "bad_output,error,match",
    [
        (3.0, TypeError, "function to return a Tensor or tuple of Tensors"),
        ((), ValueError, "output tuple must not be empty"),
        ((Tensor([1.0]), "bad"), TypeError, "tuple must contain only Tensors"),
    ],
)
def test_no_grad_recompute_keeps_output_contract(bad_output, error, match):
    x = _input()
    calls = 0

    def function(_):
        nonlocal calls
        calls += 1
        return bad_output

    with no_grad(), pytest.raises(error, match=match):
        recompute(function, x)

    assert calls == 1


def test_no_grad_recompute_returns_single_output_object_unchanged():
    x = _input()
    expected = Tensor([4.0, 5.0])

    with no_grad():
        actual = recompute(lambda _: expected, x)

    assert actual is expected


def test_no_grad_recompute_returns_tuple_objects_unchanged():
    x = _input()
    left = Tensor([4.0])
    right = Tensor([5.0])
    expected = (left, right)

    with no_grad():
        actual = recompute(lambda _: expected, x)

    assert actual is expected
    assert actual[0] is left
    assert actual[1] is right


def test_input_validation_still_happens_before_function_execution():
    calls = 0

    def function(value):
        nonlocal calls
        calls += 1
        return value

    with pytest.raises(TypeError, match="inputs must be Tensors"):
        recompute(function, object())

    assert calls == 0
