"""Cross-field contract tests for WarmupCosineScheduler learning-rate bounds."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam
from engine.scheduler import WarmupCosineScheduler
from engine.tensor import Tensor


def _optimizer(lr):
    return Adam([Tensor([0.0], requires_grad=True)], lr=lr)


def test_constructor_rejects_min_lr_above_base_learning_rate():
    optimizer = _optimizer(0.01)

    with pytest.raises(ValueError, match="min_lr must not exceed base learning rate"):
        WarmupCosineScheduler(
            optimizer,
            total_steps=10,
            warmup_steps=2,
            min_lr=0.02,
        )

    assert optimizer.lr == 0.01


def test_equal_min_and_base_learning_rates_remain_valid():
    optimizer = _optimizer(0.05)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=8,
        warmup_steps=0,
        min_lr=0.05,
    )

    assert [scheduler.get_lr(step) for step in range(12)] == [0.05] * 12


def test_load_rejects_inverted_lr_bounds_transactionally():
    optimizer = _optimizer(0.2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=20,
        warmup_steps=3,
        min_lr=0.01,
    )
    scheduler.step(6)
    before_state = scheduler.state_dict().copy()
    before_optimizer_lr = optimizer.lr

    incoming = {
        "base_lr": 0.05,
        "total_steps": 30,
        "warmup_steps": 4,
        "min_lr": 0.06,
        "last_step": 10,
    }

    with pytest.raises(
        ValueError,
        match="scheduler min_lr must not exceed base learning rate",
    ):
        scheduler.load_state_dict(incoming)

    assert scheduler.state_dict() == before_state
    assert optimizer.lr == before_optimizer_lr


def test_load_accepts_equal_lr_bound_and_restores_consistently():
    optimizer = _optimizer(0.2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=20,
        warmup_steps=3,
        min_lr=0.01,
    )
    incoming = {
        "base_lr": 0.05,
        "total_steps": 10,
        "warmup_steps": 0,
        "min_lr": 0.05,
        "last_step": 7,
    }

    scheduler.load_state_dict(incoming)

    assert scheduler.state_dict() == incoming
    assert optimizer.lr == 0.05
