import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptGraphMetadataOnWrite(np.ndarray):
    def __new__(cls, values, owner):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.owner = owner
        obj.corruptions = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.owner = getattr(source, "owner", None)
            self.corruptions = getattr(source, "corruptions", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.corruptions > 0:
            self.corruptions -= 1
            self.owner._children = (object(),)


def test_gradient_write_cannot_silently_change_leaf_graph_metadata():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = CorruptGraphMetadataOnWrite([6.0, 8.0], parameter)
    gradient_ref = parameter.grad
    gradient_before = parameter.grad.copy()

    with pytest.raises(RuntimeError, match="graph metadata changed"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient_ref
    np.testing.assert_array_equal(parameter.grad, gradient_before)
    assert parameter._children == ()


def test_plain_leaf_graph_metadata_still_allows_clipping():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0])

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert changed == 1
    np.testing.assert_allclose(parameter.grad, [0.3, 0.4], rtol=0.0, atol=1e-15)
    assert parameter._children == ()
