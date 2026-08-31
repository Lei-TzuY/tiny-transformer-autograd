import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class MutatePeerOnFlags(np.ndarray):
    """Make an ndarray metadata read observable and destructive."""

    def __new__(cls, values, peer):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.peer = peer
        obj.flags_calls = 0
        return obj

    def __array_finalize__(self, source):
        self.peer = getattr(source, "peer", None)
        self.flags_calls = getattr(source, "flags_calls", 0)

    @property
    def flags(self):
        self.flags_calls += 1
        np.asarray(self.peer)[...] = 0.5
        raise RuntimeError("untrusted ndarray flags property was dispatched")


def test_writability_preflight_does_not_dispatch_ndarray_subclass_flags():
    first = Tensor(np.array([100.0]), requires_grad=True)
    second = Tensor(np.array([100.0]), requires_grad=True)
    peer_gradient = np.array([0.25])
    gradient = MutatePeerOnFlags([10.0], peer_gradient)
    first.grad = gradient
    second.grad = peer_gradient

    changed = adaptive_clip_grad_([first, second], clip_factor=0.01)

    assert changed == 1
    assert first.grad is gradient
    assert second.grad is peer_gradient
    np.testing.assert_allclose(np.asarray(gradient), np.array([1.0]), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(peer_gradient, np.array([0.25]))
    assert gradient.flags_calls == 0
