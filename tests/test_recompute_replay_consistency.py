"""Regression tests for activation-recompute replay consistency."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.recompute import recompute
from engine.tensor import Tensor


def _rng_state_copy():
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_rejects_closed_over_parameter_drift_before_gradient_mutation():
    x = Tensor([3.0], requires_grad=True)
    weight = Tensor([2.0], requires_grad=True)
    x.grad[:] = 7.0
    weight.grad[:] = 11.0

    out = recompute(lambda inp: inp * weight, x)
    np.testing.assert_array_equal(out.data, [6.0])

    # The saved forward represented weight=2. Replaying with weight=5 would
    # silently change d(out)/d(x) from 2 to 5 without a consistency check.
    weight.data[:] = 5.0
    x_before = x.grad.copy()
    weight_before = weight.grad.copy()

    with pytest.raises(RuntimeError, match="changed values during replay"):
        ops.sum(out).backward()

    np.testing.assert_array_equal(x.grad, x_before)
    np.testing.assert_array_equal(weight.grad, weight_before)


def test_rejects_non_tensor_closure_state_drift():
    x = Tensor([1.0, 2.0], requires_grad=True)
    state = {"scale": 2.0}
    out = recompute(lambda inp: inp * state["scale"], x)

    state["scale"] = 3.0
    before = x.grad.copy()

    with pytest.raises(RuntimeError, match="changed values during replay"):
        ops.sum(out).backward()

    np.testing.assert_array_equal(x.grad, before)


def test_multi_output_replay_checks_every_output():
    x = Tensor([2.0], requires_grad=True)
    state = {"offset": 1.0}

    first, second = recompute(
        lambda inp: (inp * 2.0, inp + state["offset"]),
        x,
    )
    np.testing.assert_array_equal(first.data, [4.0])
    np.testing.assert_array_equal(second.data, [3.0])

    # Output 0 still replays exactly; output 1 does not.
    state["offset"] = 4.0
    before = x.grad.copy()

    with pytest.raises(RuntimeError, match="output 1 changed values"):
        ops.sum(first + second).backward()

    np.testing.assert_array_equal(x.grad, before)


def test_drift_failure_restores_numpy_rng_state():
    x = Tensor(np.ones(4), requires_grad=True)
    state = {"scale": 1.0}

    def section(inp):
        # Consume random numbers on both forward and replay. The replay also
        # changes value because of closure drift, forcing validation failure.
        noise = Tensor(np.random.random(inp.shape))
        return inp * noise * state["scale"]

    np.random.seed(123)
    out = recompute(section, x)
    state["scale"] = 2.0
    before = _rng_state_copy()

    with pytest.raises(RuntimeError, match="changed values during replay"):
        ops.sum(out).backward()

    after = _rng_state_copy()
    _assert_rng_equal(before, after)


def test_identical_nan_values_do_not_create_false_drift():
    x = Tensor([np.nan, 2.0], requires_grad=True)
    out = recompute(lambda inp: inp, x)

    # Consistency comparison treats a NaN replaying as NaN as unchanged. This
    # guard is about replay drift, not about imposing numerical-domain policy on
    # arbitrary user functions.
    out.backward(np.array([0.0, 1.0]))
    np.testing.assert_array_equal(x.grad, np.array([0.0, 1.0]))
