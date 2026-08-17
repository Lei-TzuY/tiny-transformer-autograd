"""Learning-rate schedulers."""

import math


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0.0):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must be between 0 and total_steps")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")

        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_step = -1

    def get_lr(self, step):
        if step < 0:
            return self.base_lr
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.base_lr if self.warmup_steps == 0 else self.min_lr
        progress = (step - self.warmup_steps) / (decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.optimizer.lr = self.get_lr(step)
        return self.optimizer.lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        base_lr = state["base_lr"]
        total_steps = state["total_steps"]
        warmup_steps = state["warmup_steps"]
        min_lr = state["min_lr"]
        last_step = state["last_step"]
        if base_lr <= 0 or total_steps <= 0:
            raise ValueError("scheduler base_lr and total_steps must be positive")
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("scheduler warmup_steps must be between 0 and total_steps")
        if min_lr < 0:
            raise ValueError("scheduler min_lr must be non-negative")

        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_step = last_step
        self.optimizer.lr = self.get_lr(self.last_step)
