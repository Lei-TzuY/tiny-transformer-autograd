"""Huge integral scheduler steps must clamp without float-conversion overflow."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam
from engine.scheduler import WarmupCosineScheduler
from engine.tensor import Tensor


HUGE_STEP = 10**400


def _make_scheduler(*, lr=0.2, total_steps=10, warmup_steps=2, min_lr=0.01):
    optimizer = Adam([Tensor([0.0], requires_grad=True)], lr=lr)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )
    return optimizer, scheduler


def test_huge_explicit_step_clamps_before_float_arithmetic():
    optimizer, scheduler = _make_scheduler()

    assert scheduler.get_lr(HUGE_STEP) == scheduler.min_lr
    assert scheduler.last_step == -1
    assert optimizer.lr == scheduler.base_lr

    assert scheduler.step(HUGE_STEP) == scheduler.min_lr
    assert scheduler.last_step == HUGE_STEP
    assert optimizer.lr == scheduler.min_lr


def test_huge_warmup_counters_form_bounded_ratio_before_float_multiply():
    warmup_steps = HUGE_STEP
    optimizer, scheduler = _make_scheduler(
        total_steps=HUGE_STEP + 8,
        warmup_steps=warmup_steps,
        min_lr=0.001,
    )
    step = HUGE_STEP // 2
    expected = scheduler.base_lr * ((step + 1) / warmup_steps)

    lr = scheduler.step(step)

    assert np.isfinite(lr)
    assert lr == expected
    assert scheduler.last_step == step
    assert optimizer.lr == expected


def test_loading_huge_last_step_restores_clamped_learning_rate():
    optimizer, scheduler = _make_scheduler()
    scheduler.step(3)

    state = scheduler.state_dict()
    state["last_step"] = HUGE_STEP
    optimizer.lr = 123.0

    scheduler.load_state_dict(state)

    assert scheduler.state_dict() == state
    assert scheduler.last_step == HUGE_STEP
    assert optimizer.lr == scheduler.min_lr


def test_single_step_schedule_preserves_existing_base_lr_contract():
    optimizer, scheduler = _make_scheduler(
        lr=0.3,
        total_steps=1,
        warmup_steps=0,
        min_lr=0.02,
    )

    assert scheduler.step(HUGE_STEP) == 0.3
    assert optimizer.lr == 0.3
    assert scheduler.last_step == HUGE_STEP
