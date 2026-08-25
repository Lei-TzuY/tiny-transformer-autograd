"""Regression tests for optimizer hyperparameters and serialized state."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


def _parameter(values=(1.0, -2.0)):
    return Tensor(values, requires_grad=True)


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
@pytest.mark.parametrize("bad_lr", [np.nan, np.inf, -np.inf, 0.0, True, "1e-3"])
def test_constructors_reject_invalid_learning_rates(optimizer_cls, bad_lr):
    with pytest.raises((TypeError, ValueError)):
        optimizer_cls([_parameter()], lr=bad_lr)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_adam_rejects_nonfinite_eps_and_weight_decay(bad):
    with pytest.raises(ValueError, match="finite"):
        Adam([_parameter()], eps=bad)
    with pytest.raises(ValueError, match="finite"):
        AdamW([_parameter()], weight_decay=bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True])
def test_sgd_rejects_invalid_weight_decay(bad):
    with pytest.raises((TypeError, ValueError)):
        SGD([_parameter()], weight_decay=bad)


def test_state_load_rejects_nonfinite_buffer_without_mutating_optimizer():
    p = _parameter()
    optimizer = Adam([p], lr=0.03, betas=(0.8, 0.9), eps=1e-6)
    p.grad[:] = [0.5, -0.25]
    optimizer.step()
    before = optimizer.state_dict()

    bad = optimizer.state_dict()
    bad["lr"] = 0.9
    bad["m"][0][0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        optimizer.load_state_dict(bad)

    after = optimizer.state_dict()
    assert after["lr"] == before["lr"]
    assert after["betas"] == before["betas"]
    assert after["eps"] == before["eps"]
    assert after["t"] == before["t"]
    np.testing.assert_array_equal(after["m"][0], before["m"][0])
    np.testing.assert_array_equal(after["v"][0], before["v"][0])


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    [
        (np.array([1.0 + 2.0j, 0.0j]), TypeError, "real numeric"),
        (np.array([object(), object()], dtype=object), TypeError, "real numeric"),
        ([0.0, 0.0], TypeError, "NumPy array"),
    ],
)
def test_state_load_rejects_non_real_or_non_array_buffers(replacement, error, message):
    optimizer = Adam([_parameter()])
    state = optimizer.state_dict()
    state["v"][0] = replacement

    with pytest.raises(error, match=message):
        optimizer.load_state_dict(state)


@pytest.mark.parametrize("bad_step", [-1, 1.5, True, np.nan])
def test_adam_state_requires_nonnegative_integer_step(bad_step):
    optimizer = Adam([_parameter()])
    state = optimizer.state_dict()
    state["t"] = bad_step

    with pytest.raises((TypeError, ValueError), match="non-negative integer"):
        optimizer.load_state_dict(state)


def test_adam_state_rejects_zero_or_nonfinite_loaded_hyperparameters():
    optimizer = Adam([_parameter()])

    for key, bad in [("lr", 0.0), ("lr", np.nan), ("eps", np.inf)]:
        state = optimizer.state_dict()
        state[key] = bad
        with pytest.raises(ValueError):
            optimizer.load_state_dict(state)


def test_adam_state_roundtrip_reproduces_the_next_update_exactly():
    first_parameter = _parameter()
    first = AdamW(
        [first_parameter],
        lr=0.02,
        betas=(0.7, 0.95),
        eps=1e-7,
        weight_decay=0.1,
    )
    first_parameter.grad[:] = [0.4, -0.8]
    first.step()

    resumed_parameter = Tensor(first_parameter.data.copy(), requires_grad=True)
    resumed = AdamW([resumed_parameter], lr=0.5)
    resumed.load_state_dict(first.state_dict())

    next_grad = np.array([-0.3, 0.6])
    first_parameter.grad[:] = next_grad
    resumed_parameter.grad[:] = next_grad
    first.step()
    resumed.step()

    np.testing.assert_array_equal(resumed_parameter.data, first_parameter.data)
    assert resumed.t == first.t
    for actual, expected in zip(resumed._m, first._m):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(resumed._v, first._v):
        np.testing.assert_array_equal(actual, expected)


def test_sgd_state_roundtrip_reproduces_momentum_update_exactly():
    first_parameter = _parameter()
    first = SGD([first_parameter], lr=0.04, momentum=0.8, weight_decay=0.03)
    first_parameter.grad[:] = [0.5, -0.7]
    first.step()

    resumed_parameter = Tensor(first_parameter.data.copy(), requires_grad=True)
    resumed = SGD([resumed_parameter], lr=0.5)
    resumed.load_state_dict(first.state_dict())

    next_grad = np.array([-0.2, 0.9])
    first_parameter.grad[:] = next_grad
    resumed_parameter.grad[:] = next_grad
    first.step()
    resumed.step()

    np.testing.assert_array_equal(resumed_parameter.data, first_parameter.data)
    np.testing.assert_array_equal(resumed._v[0], first._v[0])
