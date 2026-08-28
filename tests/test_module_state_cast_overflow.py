"""Regression tests for state-dict dtype conversion before commit."""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import ops
from engine.tensor import Tensor
from nn.module import Module


class _Pair(Module):
    def __init__(self):
        self.first = Tensor([1.0, 2.0], requires_grad=True)
        self.second = Tensor([3.0, 4.0], requires_grad=True)


class _SentinelBuffer(Module):
    def __init__(self):
        self.mask = Tensor([-np.inf, 0.0], requires_grad=False)


def _finite_wider_than_float64():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble cannot represent values beyond float64")
    value = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2.0)
    assert np.isfinite(value)
    return value


def test_cast_overflow_rejected_before_any_tensor_commit():
    module = _Pair()
    graph = ops.sum(module.first * module.first)
    first_before = module.first.data.copy()
    second_before = module.second.data.copy()
    first_version = module.first._version
    second_version = module.second._version

    state = module.state_dict()
    state["first"] = np.array([10.0, 20.0], dtype=np.float64)
    state["second"] = np.array(
        [_finite_wider_than_float64(), np.longdouble(5.0)],
        dtype=np.longdouble,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            ValueError,
            match=r"^state_dict value for second must contain only finite values$",
        ):
            module.load_state_dict(state)

    np.testing.assert_array_equal(module.first.data, first_before)
    np.testing.assert_array_equal(module.second.data, second_before)
    assert module.first._version == first_version
    assert module.second._version == second_version

    graph.backward()
    np.testing.assert_array_equal(module.first.grad, [2.0, 4.0])


def test_representable_longdouble_values_still_load():
    module = _Pair()
    state = module.state_dict()
    state["first"] = np.array([1.25, -2.5], dtype=np.longdouble)
    state["second"] = np.array([3.5, 4.75], dtype=np.longdouble)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        module.load_state_dict(state)

    assert module.first.data.dtype == np.dtype(np.float64)
    assert module.second.data.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(module.first.data, [1.25, -2.5])
    np.testing.assert_array_equal(module.second.data, [3.5, 4.75])


def test_existing_nonfinite_buffer_still_accepts_nonfinite_sentinel():
    module = _SentinelBuffer()

    module.load_state_dict(
        {"mask": np.array([-np.inf, -1.0], dtype=np.longdouble)}
    )

    assert np.isneginf(module.mask.data[0])
    assert module.mask.data[1] == -1.0


def test_finite_cast_overflow_rejected_even_for_nonfinite_buffer():
    module = _SentinelBuffer()
    before = module.mask.data.copy()
    version = module.mask._version
    state = {
        "mask": np.array(
            [_finite_wider_than_float64(), np.longdouble(0.0)],
            dtype=np.longdouble,
        )
    }

    with pytest.raises(
        ValueError,
        match=r"^state_dict value for mask must contain only finite values$",
    ):
        module.load_state_dict(state)

    np.testing.assert_array_equal(module.mask.data, before)
    assert module.mask._version == version
