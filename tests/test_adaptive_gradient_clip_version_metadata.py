import numpy as np
import pytest

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


def _clippable_parameter():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = np.array([6.0, 8.0])
    return parameter


def test_parameter_version_must_be_non_negative_python_int():
    parameter = _clippable_parameter()

    for malformed in (True, np.int64(0), "0"):
        parameter._version = malformed
        with pytest.raises(TypeError, match="parameter 0 version must be an int"):
            adaptive_clip_grad_(parameter, clip_factor=0.1)

    parameter._version = -1
    with pytest.raises(ValueError, match="parameter 0 version must be non-negative"):
        adaptive_clip_grad_(parameter, clip_factor=0.1)


def test_late_malformed_version_is_rejected_before_earlier_gradient_write():
    first = _clippable_parameter()
    second = _clippable_parameter()
    first_grad = first.grad
    first_before = first.grad.copy()
    second._version = -1

    with pytest.raises(ValueError, match="parameter 1 version must be non-negative"):
        adaptive_clip_grad_([first, second], clip_factor=0.1)

    assert first.grad is first_grad
    np.testing.assert_array_equal(first.grad, first_before)


def test_valid_version_metadata_keeps_normal_clipping_behavior():
    parameter = _clippable_parameter()
    version = parameter._version

    assert isinstance(version, int) and not isinstance(version, bool) and version >= 0
    assert adaptive_clip_grad_(parameter, clip_factor=0.1) == 1
    np.testing.assert_allclose(parameter.grad, [0.3, 0.4])
    assert parameter._version == version
