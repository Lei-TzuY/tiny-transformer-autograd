"""Regression tests for validating EMA decay changes after construction."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.ema import ExponentialMovingAverage
from engine.tensor import Tensor


def _assert_state_equal(first, second):
    assert first["decay"] == second["decay"]
    assert first["num_updates"] == second["num_updates"]
    assert len(first["averages"]) == len(second["averages"])
    for left, right in zip(first["averages"], second["averages"]):
        np.testing.assert_array_equal(left, right)


def test_decay_can_be_changed_with_the_same_public_validation_contract():
    parameter = Tensor([2.0])
    ema = ExponentialMovingAverage(parameter, decay=0.75)

    ema.decay = np.float64(0.5)
    parameter.data[...] = [6.0]
    ema.update()

    assert ema.decay == 0.5
    assert ema.num_updates == 1
    np.testing.assert_array_equal(ema.averages()[0], [4.0])


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (True, TypeError, "EMA decay must be a real number"),
        ("0.5", TypeError, "EMA decay must be a real number"),
        (np.inf, ValueError, "EMA decay must be finite"),
        (10**400, ValueError, "EMA decay must be finite"),
        (-0.1, ValueError, r"EMA decay must be in \[0, 1\]"),
        (1.1, ValueError, r"EMA decay must be in \[0, 1\]"),
    ],
)
def test_invalid_decay_reassignment_is_transactional(value, error_type, message):
    parameter = Tensor([3.0], requires_grad=True)
    parameter.grad[...] = [7.0]
    ema = ExponentialMovingAverage(parameter, decay=0.8)
    state_before = ema.state_dict()
    data_before = parameter.data.copy()
    grad_before = parameter.grad.copy()
    version_before = parameter._version
    rng_before = np.random.get_state()

    with pytest.raises(error_type, match=message):
        ema.decay = value

    _assert_state_equal(ema.state_dict(), state_before)
    np.testing.assert_array_equal(parameter.data, data_before)
    np.testing.assert_array_equal(parameter.grad, grad_before)
    assert parameter._version == version_before

    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_decay_endpoints_remain_assignable_after_construction():
    parameter = Tensor([1.0])
    ema = ExponentialMovingAverage(parameter, decay=0.5)

    ema.decay = 0
    assert ema.decay == 0.0
    parameter.data[...] = [5.0]
    ema.update()
    np.testing.assert_array_equal(ema.averages()[0], [5.0])

    ema.decay = 1
    assert ema.decay == 1.0
    parameter.data[...] = [9.0]
    ema.update()
    np.testing.assert_array_equal(ema.averages()[0], [5.0])
    assert ema.num_updates == 2
