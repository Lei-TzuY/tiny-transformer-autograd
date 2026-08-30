import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class LyingVersion(int):
    """Integer subclass whose comparisons must not participate in version validation."""

    comparisons = 0

    def __new__(cls, value):
        return super().__new__(cls, value)

    def __lt__(self, other):
        type(self).comparisons += 1
        return False

    def __ne__(self, other):
        type(self).comparisons += 1
        return False


def test_version_int_subclass_is_rejected_without_dispatching_comparison():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0], dtype=np.float64)
    parameter.grad = gradient
    parameter._version = LyingVersion(0)
    LyingVersion.comparisons = 0

    with pytest.raises(TypeError, match="version must be an int"):
        adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert LyingVersion.comparisons == 0
    np.testing.assert_array_equal(gradient, np.array([6.0, 8.0]))


def test_builtin_version_int_keeps_existing_clipping_semantics():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([6.0, 8.0], dtype=np.float64)
    parameter.grad = gradient

    changed = adaptive_clip_grad_(parameter, clip_factor=0.1, eps=1e-3)

    assert changed == 1
    assert type(parameter._version) is int
    np.testing.assert_allclose(gradient, np.array([0.3, 0.4]), rtol=0.0, atol=1e-15)
