"""Validation and restore tests for WarmupCosineScheduler."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam
from engine.scheduler import WarmupCosineScheduler
from engine.tensor import Tensor


def _make_scheduler(*, lr=1e-3, total=20, warmup=2, min_lr=1e-5):
    optimizer = Adam([Tensor([0.0], requires_grad=True)], lr=lr)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total,
        warmup_steps=warmup,
        min_lr=min_lr,
    )
    return optimizer, scheduler


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"total_steps": 0}, ValueError, "total_steps"),
        ({"total_steps": 3.5}, TypeError, "total_steps must be an integer"),
        ({"total_steps": True}, TypeError, "total_steps must be an integer"),
        ({"warmup_steps": 1.5}, TypeError, "warmup_steps must be an integer"),
        ({"warmup_steps": True}, TypeError, "warmup_steps must be an integer"),
        ({"warmup_steps": 21}, ValueError, "between 0 and total_steps"),
        ({"min_lr": np.nan}, ValueError, "min_lr must be finite"),
        ({"min_lr": np.inf}, ValueError, "min_lr must be finite"),
        ({"min_lr": True}, TypeError, "min_lr must be a real number"),
        ({"min_lr": -1e-3}, ValueError, "min_lr must be non-negative"),
    ],
)
def test_constructor_rejects_invalid_schedule_arguments(kwargs, error, message):
    optimizer = Adam([Tensor([0.0], requires_grad=True)], lr=1e-3)
    defaults = {"total_steps": 20, "warmup_steps": 2, "min_lr": 1e-5}
    defaults.update(kwargs)

    with pytest.raises(error, match=message):
        WarmupCosineScheduler(optimizer, **defaults)


@pytest.mark.parametrize("bad_lr", [np.nan, np.inf, -np.inf, 0.0, -1.0, True])
def test_constructor_rejects_invalid_optimizer_learning_rate(bad_lr):
    class DummyOptimizer:
        lr = bad_lr

    with pytest.raises((TypeError, ValueError), match="optimizer.lr"):
        WarmupCosineScheduler(DummyOptimizer(), total_steps=10)


@pytest.mark.parametrize(
    ("method", "step", "error"),
    [
        ("get_lr", 1.5, TypeError),
        ("get_lr", True, TypeError),
        ("get_lr", -2, ValueError),
        ("step", 1.5, TypeError),
        ("step", True, TypeError),
        ("step", -1, ValueError),
    ],
)
def test_step_api_rejects_non_integral_or_invalid_steps(method, step, error):
    _, scheduler = _make_scheduler()

    with pytest.raises(error, match="step"):
        getattr(scheduler, method)(step)


def test_implicit_step_sequence_is_unchanged():
    _, implicit = _make_scheduler(total=8, warmup=2, min_lr=1e-4)
    _, explicit = _make_scheduler(total=8, warmup=2, min_lr=1e-4)

    implicit_lrs = [implicit.step() for _ in range(10)]
    explicit_lrs = [explicit.step(step) for step in range(10)]

    np.testing.assert_array_equal(implicit_lrs, explicit_lrs)
    assert implicit.last_step == explicit.last_step == 9


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("base_lr", np.nan, ValueError, "base_lr.*finite"),
        ("base_lr", np.inf, ValueError, "base_lr.*finite"),
        ("base_lr", True, TypeError, "base_lr.*real number"),
        ("base_lr", 0.0, ValueError, "base_lr.*positive"),
        ("total_steps", 8.5, TypeError, "total_steps.*integer"),
        ("total_steps", True, TypeError, "total_steps.*integer"),
        ("total_steps", 0, ValueError, "total_steps"),
        ("warmup_steps", 1.5, TypeError, "warmup_steps.*integer"),
        ("warmup_steps", True, TypeError, "warmup_steps.*integer"),
        ("warmup_steps", 99, ValueError, "between 0 and total_steps"),
        ("min_lr", np.nan, ValueError, "min_lr.*finite"),
        ("min_lr", np.inf, ValueError, "min_lr.*finite"),
        ("min_lr", True, TypeError, "min_lr.*real number"),
        ("min_lr", -0.1, ValueError, "min_lr.*non-negative"),
        ("last_step", 2.5, TypeError, "last_step.*integer"),
        ("last_step", True, TypeError, "last_step.*integer"),
        ("last_step", -2, ValueError, "last_step"),
    ],
)
def test_load_rejects_malformed_fields_without_partial_restore(
    field, value, error, message
):
    optimizer, scheduler = _make_scheduler(total=20, warmup=2, min_lr=1e-5)
    scheduler.step(4)
    before_state = scheduler.state_dict().copy()
    before_optimizer_lr = optimizer.lr

    incoming = {
        "base_lr": 0.2,
        "total_steps": 30,
        "warmup_steps": 3,
        "min_lr": 0.01,
        "last_step": 7,
    }
    incoming[field] = value

    with pytest.raises(error, match=message):
        scheduler.load_state_dict(incoming)

    assert scheduler.state_dict() == before_state
    assert optimizer.lr == before_optimizer_lr


def test_load_requires_mapping_and_all_serialized_fields():
    _, scheduler = _make_scheduler()

    with pytest.raises(TypeError, match="state must be a mapping"):
        scheduler.load_state_dict(None)

    state = scheduler.state_dict()
    state.pop("last_step")
    with pytest.raises(ValueError, match="missing keys.*last_step"):
        scheduler.load_state_dict(state)


def test_extra_checkpoint_keys_are_ignored_for_forward_compatibility():
    _, scheduler = _make_scheduler()
    state = scheduler.state_dict()
    state["future_metadata"] = {"version": 2}

    scheduler.load_state_dict(state)

    assert scheduler.state_dict()["last_step"] == -1


def test_unstepped_state_restores_base_learning_rate():
    optimizer, scheduler = _make_scheduler(lr=0.25, total=10, warmup=2)
    state = scheduler.state_dict()
    optimizer.lr = 999.0

    scheduler.load_state_dict(state)

    assert scheduler.last_step == -1
    assert optimizer.lr == 0.25


def test_state_roundtrip_reproduces_all_future_learning_rates_exactly():
    optimizer, scheduler = _make_scheduler(
        lr=0.2,
        total=17,
        warmup=4,
        min_lr=0.003,
    )
    for step in range(7):
        scheduler.step(step)
    state = scheduler.state_dict()

    restored_optimizer, restored = _make_scheduler(
        lr=0.9,
        total=3,
        warmup=0,
        min_lr=0.0,
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored_optimizer.lr == scheduler.get_lr(state["last_step"])
    for step in range(7, 25):
        assert restored.step(step) == scheduler.step(step)
