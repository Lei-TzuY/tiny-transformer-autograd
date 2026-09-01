"""Gradient centralization preserves canonical explicit-leaf provenance."""

import numpy as np
import pytest

from engine import Tensor, centralize_gradients_
from engine.tensor import _no_backward


class CorruptLeafProvenanceOnWrite(np.ndarray):
    def __new__(cls, values, parameter):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.parameter = parameter
        obj.corruptions = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.parameter = getattr(source, "parameter", None)
            self.corruptions = getattr(source, "corruptions", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.corruptions > 0:
            self.corruptions -= 1
            self.parameter._children = (object(),)
            self.parameter._backward_fn = lambda: None
            self.parameter._detached_by_no_grad = True


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda parameter: setattr(parameter, "_children", []), "graph metadata"),
        (
            lambda parameter: setattr(parameter, "_backward_fn", lambda: None),
            "backward metadata",
        ),
        (
            lambda parameter: setattr(parameter, "_detached_by_no_grad", True),
            "detached provenance",
        ),
    ],
)
def test_centralization_rejects_noncanonical_leaf_provenance_before_write(mutate, error):
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    gradient = parameter.grad
    gradient[...] = np.array([[3.0, 1.0]])
    before = gradient.copy()
    mutate(parameter)

    with pytest.raises(TypeError, match=error):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)


def test_gradient_write_cannot_silently_corrupt_leaf_provenance():
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    gradient = CorruptLeafProvenanceOnWrite([[3.0, 1.0]], parameter)
    parameter.grad = gradient
    before = gradient.copy()

    with pytest.raises(RuntimeError, match="leaf provenance changed"):
        centralize_gradients_(parameter)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, before)
    assert parameter._children == ()
    assert parameter._backward_fn is _no_backward
    assert parameter._detached_by_no_grad is False


def test_plain_explicit_leaf_provenance_remains_accepted():
    parameter = Tensor([[1.0, 2.0]], requires_grad=True)
    gradient = parameter.grad
    gradient[...] = np.array([[3.0, 1.0]])

    changed = centralize_gradients_(parameter)

    assert changed == 1
    assert parameter.grad is gradient
    np.testing.assert_array_equal(gradient, np.array([[1.0, -1.0]]))
    assert parameter._children == ()
    assert parameter._backward_fn is _no_backward
    assert parameter._detached_by_no_grad is False
