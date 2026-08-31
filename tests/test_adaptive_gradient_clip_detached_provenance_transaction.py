import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class CorruptDetachedProvenanceOnWrite(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.target = None
        obj.changes_remaining = 1
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.target = getattr(source, "target", None)
            self.changes_remaining = getattr(source, "changes_remaining", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.changes_remaining <= 0:
            return
        self.changes_remaining -= 1
        self.target._detached_by_no_grad = True


def test_commit_detects_and_restores_leaf_detached_provenance():
    active = Tensor([3.0, 4.0], requires_grad=True)
    frozen = Tensor([5.0], requires_grad=False)
    gradient = CorruptDetachedProvenanceOnWrite([6.0, 8.0])
    gradient.target = frozen
    active.grad = gradient
    grad_before = gradient.copy()

    assert frozen._detached_by_no_grad is False
    frozen.backward()

    with pytest.raises(RuntimeError, match="detached provenance changed for parameter 1"):
        adaptive_clip_grad_([active, frozen], clip_factor=0.1)

    assert active.grad is gradient
    np.testing.assert_array_equal(active.grad, grad_before)
    assert frozen._detached_by_no_grad is False
    frozen.backward()


def test_rejects_malformed_leaf_detached_provenance_before_gradient_write():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0])
    parameter.grad = gradient
    parameter._detached_by_no_grad = True
    grad_before = gradient.copy()

    with pytest.raises(TypeError, match="detached provenance must be false"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, grad_before)
