"""Regression coverage for scheduler public numeric/interface validation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.scheduler import WarmupCosineScheduler


class _DummyOptimizer:
    def __init__(self, lr):
        self.lr = lr


def test_constructor_rejects_optimizer_without_learning_rate_attribute():
    with pytest.raises(TypeError, match="optimizer must expose an lr attribute"):
        WarmupCosineScheduler(object(), total_steps=10)


def test_constructor_normalizes_unrepresentable_optimizer_lr():
    optimizer = _DummyOptimizer(10**400)

    with pytest.raises(ValueError, match=r"optimizer\.lr must be finite"):
        WarmupCosineScheduler(optimizer, total_steps=10)


def test_constructor_normalizes_unrepresentable_min_lr():
    optimizer = _DummyOptimizer(1.0)

    with pytest.raises(ValueError, match="min_lr must be finite"):
        WarmupCosineScheduler(optimizer, total_steps=10, min_lr=10**400)


def test_load_normalizes_unrepresentable_base_lr_transactionally():
    optimizer = _DummyOptimizer(0.2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        min_lr=0.01,
    )
    scheduler.step(3)
    before_state = scheduler.state_dict().copy()
    before_lr = optimizer.lr
    incoming = before_state.copy()
    incoming["base_lr"] = 10**400

    with pytest.raises(ValueError, match="scheduler base_lr must be finite"):
        scheduler.load_state_dict(incoming)

    assert scheduler.state_dict() == before_state
    assert optimizer.lr == before_lr


def test_load_normalizes_unrepresentable_min_lr_transactionally():
    optimizer = _DummyOptimizer(0.2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        min_lr=0.01,
    )
    scheduler.step(3)
    before_state = scheduler.state_dict().copy()
    before_lr = optimizer.lr
    incoming = before_state.copy()
    incoming["min_lr"] = 10**400

    with pytest.raises(ValueError, match="scheduler min_lr must be finite"):
        scheduler.load_state_dict(incoming)

    assert scheduler.state_dict() == before_state
    assert optimizer.lr == before_lr


def test_large_representable_real_values_remain_supported():
    optimizer = _DummyOptimizer(10**300)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=np.int64(4),
        warmup_steps=np.int64(1),
        min_lr=10**299,
    )

    assert scheduler.base_lr == float(10**300)
    assert scheduler.min_lr == float(10**299)
    assert scheduler.step(0) == float(10**300)
    assert np.isfinite(scheduler.step(1))
