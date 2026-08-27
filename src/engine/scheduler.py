"""Learning-rate schedulers."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0.0):
        base_lr = _positive_finite_real("optimizer.lr", optimizer.lr)
        total_steps = _positive_integer("total_steps", total_steps)
        warmup_steps = _integer("warmup_steps", warmup_steps, minimum=0)
        if warmup_steps > total_steps:
            raise ValueError("warmup_steps must be between 0 and total_steps")
        min_lr = _non_negative_finite_real("min_lr", min_lr)
        _validate_min_lr(base_lr, min_lr, name="min_lr")

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_step = -1

    def get_lr(self, step):
        step = _integer("step", step, minimum=-1)
        return _schedule_lr(
            self.base_lr,
            self.total_steps,
            self.warmup_steps,
            self.min_lr,
            step,
        )

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        else:
            step = _integer("step", step, minimum=0)
        lr = self.get_lr(step)
        self.optimizer.lr = lr
        self.last_step = step
        return lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        """Restore scheduler state after validating the complete checkpoint.

        Validation happens before mutating either the scheduler or its optimizer,
        so a malformed late field cannot leave a partially restored training
        state. Extra keys are tolerated for forward-compatible metadata, while
        all fields emitted by :meth:`state_dict` remain required.
        """
        if not isinstance(state, Mapping):
            raise TypeError("scheduler state must be a mapping")

        required = {
            "base_lr",
            "total_steps",
            "warmup_steps",
            "min_lr",
            "last_step",
        }
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"scheduler state missing keys: {missing}")

        base_lr = _positive_finite_real("scheduler base_lr", state["base_lr"])
        total_steps = _positive_integer("scheduler total_steps", state["total_steps"])
        warmup_steps = _integer(
            "scheduler warmup_steps", state["warmup_steps"], minimum=0
        )
        if warmup_steps > total_steps:
            raise ValueError(
                "scheduler warmup_steps must be between 0 and total_steps"
            )
        min_lr = _non_negative_finite_real("scheduler min_lr", state["min_lr"])
        _validate_min_lr(base_lr, min_lr, name="scheduler min_lr")
        last_step = _integer("scheduler last_step", state["last_step"], minimum=-1)

        # Compute the restored optimizer LR from validated local values before
        # mutating self. This keeps load_state_dict transactional even if the
        # schedule arithmetic is changed later.
        restored_lr = _schedule_lr(
            base_lr,
            total_steps,
            warmup_steps,
            min_lr,
            last_step,
        )

        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_step = last_step
        self.optimizer.lr = restored_lr


def _schedule_lr(base_lr, total_steps, warmup_steps, min_lr, step):
    if step < 0:
        return base_lr
    if warmup_steps and step < warmup_steps:
        # Divide the integral step counters before multiplying by a float. Both
        # integers may be arbitrarily large, while their ratio is guaranteed to
        # be in (0, 1] and therefore representable without integer->float overflow.
        return base_lr * ((step + 1) / warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return base_lr if warmup_steps == 0 else min_lr

    # The schedule is already clamped after its final decay step. Check that
    # integral boundary before forming a floating-point quotient so enormous
    # explicit/checkpoint step counters cannot overflow while computing a value
    # that is known to be exactly min_lr.
    if step >= total_steps - 1:
        return min_lr

    progress = (step - warmup_steps) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _integer(name, value, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        comparator = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return value


def _positive_integer(name, value):
    value = _integer(name, value, minimum=1)
    return value


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_finite_real(name, value):
    value = _finite_real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_finite_real(name, value):
    value = _finite_real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_min_lr(base_lr, min_lr, *, name):
    if min_lr > base_lr:
        raise ValueError(f"{name} must not exceed base learning rate")
