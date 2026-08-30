import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutatePeerOnShape(np.ndarray):
    """Make an ndarray shape metadata read observable and destructive."""

    def __new__(cls, values, peer):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.peer = peer
        obj.shape_calls = 0
        return obj

    def __array_finalize__(self, source):
        self.peer = getattr(source, "peer", None)
        self.shape_calls = getattr(source, "shape_calls", 0)

    @property
    def shape(self):
        self.shape_calls += 1
        np.asarray(self.peer)[...] = 0.5
        raise RuntimeError("untrusted ndarray shape property was dispatched")


def test_parameter_shape_validation_does_not_dispatch_ndarray_subclass_shape():
    first = Tensor(np.array([100.0]), requires_grad=True)
    second = Tensor(np.array([100.0]), requires_grad=True)
    peer_gradient = np.array([0.25])
    hostile_data = MutatePeerOnShape([100.0], peer_gradient)
    first._data = hostile_data
    first.grad = np.array([10.0])
    second.grad = peer_gradient

    changed = adaptive_clip_grad_([first, second], clip_factor=0.01)

    assert changed == 1
    assert first.data is hostile_data
    assert second.grad is peer_gradient
    np.testing.assert_allclose(first.grad, np.array([1.0]), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(peer_gradient, np.array([0.25]))
    assert hostile_data.shape_calls == 0
