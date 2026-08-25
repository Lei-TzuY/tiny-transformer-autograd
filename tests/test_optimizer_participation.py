"""Regression tests for optimizer behavior when parameters skip updates."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.optim import Adam, AdamW
from engine.tensor import Tensor


def _loss_for(parameter, scale):
    scale_tensor = Tensor(np.full(parameter.shape, scale))
    return ops.sum(ops.mul(parameter, scale_tensor))


def _adam_kwargs():
    return {
        "lr": 0.05,
        "betas": (0.7, 0.8),
        "eps": 1e-9,
        "weight_decay": 0.0,
    }


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_delayed_parameter_uses_its_own_bias_correction_step(optimizer_cls):
    first = Tensor([1.0, -1.0], requires_grad=True)
    delayed = Tensor([2.0, -3.0], requires_grad=True)
    optimizer = optimizer_cls([first, delayed], **_adam_kwargs())

    optimizer.zero_grad(set_to_none=True)
    _loss_for(first, 2.0).backward()
    assert delayed.grad is None
    optimizer.step()
    assert optimizer.t == 1
    assert optimizer._steps == [1, 0]

    reference_parameter = Tensor(delayed.data.copy(), requires_grad=True)
    reference = optimizer_cls([reference_parameter], **_adam_kwargs())

    optimizer.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    _loss_for(delayed, -3.0).backward()
    _loss_for(reference_parameter, -3.0).backward()
    assert first.grad is None

    optimizer.step()
    reference.step()

    np.testing.assert_array_equal(delayed.data, reference_parameter.data)
    np.testing.assert_array_equal(optimizer._m[1], reference._m[0])
    np.testing.assert_array_equal(optimizer._v[1], reference._v[0])
    assert optimizer.t == 2
    assert optimizer._steps == [1, 1]
    assert reference._steps == [1]


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_frozen_then_unfrozen_parameter_starts_at_first_adam_step(optimizer_cls):
    parameter = Tensor([1.5, -0.5], requires_grad=False)
    optimizer = optimizer_cls([parameter], **_adam_kwargs())

    for _ in range(3):
        optimizer.step()
    assert optimizer.t == 3
    assert optimizer._steps == [0]
    np.testing.assert_array_equal(parameter.data, np.array([1.5, -0.5]))

    parameter.requires_grad = True
    optimizer.zero_grad(set_to_none=True)
    _loss_for(parameter, 4.0).backward()

    reference_parameter = Tensor([1.5, -0.5], requires_grad=True)
    reference = optimizer_cls([reference_parameter], **_adam_kwargs())
    reference.zero_grad(set_to_none=True)
    _loss_for(reference_parameter, 4.0).backward()

    optimizer.step()
    reference.step()

    np.testing.assert_array_equal(parameter.data, reference_parameter.data)
    assert optimizer.t == 4
    assert optimizer._steps == [1]


def test_set_to_none_allows_autograd_to_mark_only_participating_parameters():
    used = Tensor([1.0, 2.0], requires_grad=True)
    unused = Tensor([3.0, 4.0], requires_grad=True)
    optimizer = Adam([used, unused])

    optimizer.zero_grad(set_to_none=True)
    assert used.grad is None
    assert unused.grad is None

    _loss_for(used, 2.5).backward()
    np.testing.assert_array_equal(used.grad, np.array([2.5, 2.5]))
    assert unused.grad is None

    optimizer.step()
    assert optimizer._steps == [1, 0]


def test_zero_grad_default_keeps_existing_zero_buffer_behavior():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad[:] = [3.0, -4.0]
    optimizer = Adam([parameter])

    optimizer.zero_grad()

    np.testing.assert_array_equal(parameter.grad, np.zeros(2))


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_zero_grad_requires_boolean_set_to_none(bad):
    optimizer = Adam([Tensor([1.0], requires_grad=True)])
    with pytest.raises(TypeError, match="boolean"):
        optimizer.zero_grad(set_to_none=bad)


def test_heterogeneous_parameter_steps_roundtrip_exactly():
    first = Tensor([1.0, -1.0], requires_grad=True)
    second = Tensor([2.0, -2.0], requires_grad=True)
    optimizer = AdamW(
        [first, second],
        lr=0.05,
        betas=(0.7, 0.8),
        eps=1e-9,
        weight_decay=0.1,
    )

    optimizer.zero_grad(set_to_none=True)
    _loss_for(first, 1.5).backward()
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    _loss_for(first, -0.25).backward()
    _loss_for(second, 0.75).backward()
    optimizer.step()

    state = optimizer.state_dict()
    assert state["t"] == 2
    assert state["steps"] == [2, 1]

    resumed_first = Tensor(first.data.copy(), requires_grad=True)
    resumed_second = Tensor(second.data.copy(), requires_grad=True)
    resumed = AdamW([resumed_first, resumed_second], lr=0.9)
    resumed.load_state_dict(state)
    assert resumed._steps == [2, 1]

    optimizer.zero_grad(set_to_none=True)
    resumed.zero_grad(set_to_none=True)
    _loss_for(second, -1.25).backward()
    _loss_for(resumed_second, -1.25).backward()
    optimizer.step()
    resumed.step()

    np.testing.assert_array_equal(resumed_first.data, first.data)
    np.testing.assert_array_equal(resumed_second.data, second.data)
    assert resumed.t == optimizer.t
    assert resumed._steps == optimizer._steps
    for actual, expected in zip(resumed._m, optimizer._m):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(resumed._v, optimizer._v):
        np.testing.assert_array_equal(actual, expected)


def test_legacy_scalar_step_state_migrates_to_all_parameter_steps():
    parameters = [
        Tensor([1.0], requires_grad=True),
        Tensor([2.0], requires_grad=True),
    ]
    optimizer = Adam(parameters)
    state = optimizer.state_dict()
    state["t"] = 4
    del state["steps"]

    optimizer.load_state_dict(state)

    assert optimizer.t == 4
    assert optimizer._steps == [4, 4]
    assert optimizer.state_dict()["steps"] == [4, 4]


@pytest.mark.parametrize(
    "bad_steps",
    [
        [0],
        [0, 2],
        [0, -1],
        [0, 1.5],
        [0, True],
        "0,1",
    ],
)
def test_invalid_parameter_step_state_is_rejected_transactionally(bad_steps):
    parameters = [
        Tensor([1.0], requires_grad=True),
        Tensor([2.0], requires_grad=True),
    ]
    optimizer = Adam(parameters, lr=0.03)
    optimizer.zero_grad(set_to_none=True)
    _loss_for(parameters[0], 0.5).backward()
    optimizer.step()
    before = optimizer.state_dict()

    bad = optimizer.state_dict()
    bad["lr"] = 0.8
    bad["steps"] = bad_steps

    with pytest.raises((TypeError, ValueError)):
        optimizer.load_state_dict(bad)

    after = optimizer.state_dict()
    assert after["lr"] == before["lr"]
    assert after["t"] == before["t"]
    assert after["steps"] == before["steps"]
    for actual, expected in zip(after["m"], before["m"]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(after["v"], before["v"]):
        np.testing.assert_array_equal(actual, expected)
