"""Metric-driven learning-rate reduction for validation plateaus."""

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Integral, Real
import threading

import numpy as np


_FORMAT_VERSION = 1
_SCHEDULER_NAME = "ReduceLROnPlateau"
_MIN_POSITIVE_LR = float(np.finfo(np.float64).smallest_subnormal)


class ReduceLROnPlateau:
    """Reduce an optimizer learning rate when a monitored metric stalls.

    ``patience`` is the number of consecutive non-improving observations that
    are tolerated. A reduction is attempted on the next one, so ``patience=0``
    reduces on the first non-improving metric after a baseline has been set.

    Relative thresholds are measured against ``abs(best)``. Improvement
    comparisons use exact rational arithmetic over the already-validated
    binary64 values, avoiding subtraction/product overflow for extreme finite
    metrics.
    """

    def __init__(
        self,
        optimizer,
        *,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.0,
        eps=1e-8,
    ):
        current_lr = _optimizer_lr(optimizer)
        mode = _choice("mode", mode, ("min", "max"))
        factor = _factor(factor)
        patience = _nonnegative_integer("patience", patience)
        threshold = _nonnegative_finite_real("threshold", threshold)
        threshold_mode = _choice(
            "threshold_mode", threshold_mode, ("rel", "abs")
        )
        cooldown = _nonnegative_integer("cooldown", cooldown)
        min_lr = _nonnegative_finite_real("min_lr", min_lr)
        if min_lr > current_lr:
            raise ValueError("min_lr must not exceed optimizer.lr")
        eps = _nonnegative_finite_real("eps", eps)

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.step_count = 0
        self.reductions = 0
        self._lock = threading.RLock()

    @property
    def in_cooldown(self):
        with self._lock:
            return self.cooldown_counter > 0

    def get_lr(self):
        """Return the validated current learning rate of the bound optimizer."""
        with self._lock:
            current_lr = _optimizer_lr(self.optimizer)
            self._validate_lr_floor(current_lr)
            return current_lr

    def step(self, metric):
        """Observe one metric value, possibly reduce LR, and return current LR."""
        metric = _finite_real("metric", metric)

        with self._lock:
            current_lr = _optimizer_lr(self.optimizer)
            self._validate_lr_floor(current_lr)

            best = self.best
            num_bad_epochs = self.num_bad_epochs
            cooldown_counter = self.cooldown_counter
            step_count = self.step_count + 1
            reductions = self.reductions

            if best is None:
                best = metric
                num_bad_epochs = 0
            else:
                improved = _is_better(
                    metric,
                    best,
                    mode=self.mode,
                    threshold=self.threshold,
                    threshold_mode=self.threshold_mode,
                )

                was_in_cooldown = cooldown_counter > 0
                if was_in_cooldown:
                    cooldown_counter -= 1

                if improved:
                    best = metric
                    num_bad_epochs = 0
                elif was_in_cooldown:
                    # Observations consumed while cooling down can still update
                    # ``best`` above, but they do not accumulate bad epochs.
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

            new_lr = current_lr
            should_reduce = best is not None and num_bad_epochs > self.patience
            if should_reduce:
                candidate_lr = _reduced_lr(current_lr, self.factor, self.min_lr)
                reduction = current_lr - candidate_lr
                if reduction > self.eps:
                    _assign_optimizer_lr(
                        self.optimizer,
                        candidate_lr,
                        previous_lr=current_lr,
                    )
                    new_lr = candidate_lr
                    reductions += 1
                    cooldown_counter = self.cooldown
                num_bad_epochs = 0

            self.best = best
            self.num_bad_epochs = num_bad_epochs
            self.cooldown_counter = cooldown_counter
            self.step_count = step_count
            self.reductions = reductions
            return new_lr

    def state_dict(self):
        """Return a JSON-friendly independent snapshot of scheduler state."""
        with self._lock:
            current_lr = _optimizer_lr(self.optimizer)
            self._validate_lr_floor(current_lr)
            return {
                "format_version": _FORMAT_VERSION,
                "scheduler": _SCHEDULER_NAME,
                "mode": self.mode,
                "factor": self.factor,
                "patience": self.patience,
                "threshold": self.threshold,
                "threshold_mode": self.threshold_mode,
                "cooldown": self.cooldown,
                "min_lr": self.min_lr,
                "eps": self.eps,
                "best": self.best,
                "num_bad_epochs": self.num_bad_epochs,
                "cooldown_counter": self.cooldown_counter,
                "step_count": self.step_count,
                "reductions": self.reductions,
                "current_lr": current_lr,
            }

    def load_state_dict(self, state):
        """Restore scheduler and optimizer LR after full-envelope validation."""
        if not isinstance(state, Mapping):
            raise TypeError("plateau scheduler state must be a mapping")

        required = {
            "format_version",
            "scheduler",
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
            "best",
            "num_bad_epochs",
            "cooldown_counter",
            "step_count",
            "reductions",
            "current_lr",
        }
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"plateau scheduler state missing keys: {missing}")

        format_version = _nonnegative_integer(
            "plateau scheduler format_version", state["format_version"]
        )
        if format_version != _FORMAT_VERSION:
            raise ValueError(
                f"plateau scheduler format_version must be {_FORMAT_VERSION}"
            )
        if state["scheduler"] != _SCHEDULER_NAME:
            raise ValueError(
                f"plateau scheduler scheduler must be '{_SCHEDULER_NAME}'"
            )

        mode = _choice("plateau scheduler mode", state["mode"], ("min", "max"))
        factor = _factor(state["factor"], name="plateau scheduler factor")
        patience = _nonnegative_integer(
            "plateau scheduler patience", state["patience"]
        )
        threshold = _nonnegative_finite_real(
            "plateau scheduler threshold", state["threshold"]
        )
        threshold_mode = _choice(
            "plateau scheduler threshold_mode",
            state["threshold_mode"],
            ("rel", "abs"),
        )
        cooldown = _nonnegative_integer(
            "plateau scheduler cooldown", state["cooldown"]
        )
        min_lr = _nonnegative_finite_real(
            "plateau scheduler min_lr", state["min_lr"]
        )
        eps = _nonnegative_finite_real("plateau scheduler eps", state["eps"])
        current_lr = _positive_finite_real(
            "plateau scheduler current_lr", state["current_lr"]
        )
        if current_lr < min_lr:
            raise ValueError("plateau scheduler current_lr must be at least min_lr")

        best_value = state["best"]
        if best_value is None:
            best = None
        else:
            best = _finite_real("plateau scheduler best", best_value)

        num_bad_epochs = _nonnegative_integer(
            "plateau scheduler num_bad_epochs", state["num_bad_epochs"]
        )
        cooldown_counter = _nonnegative_integer(
            "plateau scheduler cooldown_counter", state["cooldown_counter"]
        )
        step_count = _nonnegative_integer(
            "plateau scheduler step_count", state["step_count"]
        )
        reductions = _nonnegative_integer(
            "plateau scheduler reductions", state["reductions"]
        )

        if num_bad_epochs > patience:
            raise ValueError(
                "plateau scheduler num_bad_epochs must not exceed patience"
            )
        if cooldown_counter > cooldown:
            raise ValueError(
                "plateau scheduler cooldown_counter must not exceed cooldown"
            )
        if reductions > step_count:
            raise ValueError(
                "plateau scheduler reductions must not exceed step_count"
            )
        if best is None:
            if step_count != 0:
                raise ValueError(
                    "plateau scheduler best may be None only when step_count is zero"
                )
            if num_bad_epochs != 0 or cooldown_counter != 0 or reductions != 0:
                raise ValueError(
                    "empty plateau scheduler state must have zero counters"
                )
        elif step_count == 0:
            raise ValueError(
                "plateau scheduler best requires a positive step_count"
            )
        if cooldown_counter > 0 and reductions == 0:
            raise ValueError(
                "plateau scheduler cooldown requires at least one reduction"
            )

        with self._lock:
            previous_lr = _optimizer_lr(self.optimizer)
            _assign_optimizer_lr(
                self.optimizer,
                current_lr,
                previous_lr=previous_lr,
            )

            self.mode = mode
            self.factor = factor
            self.patience = patience
            self.threshold = threshold
            self.threshold_mode = threshold_mode
            self.cooldown = cooldown
            self.min_lr = min_lr
            self.eps = eps
            self.best = best
            self.num_bad_epochs = num_bad_epochs
            self.cooldown_counter = cooldown_counter
            self.step_count = step_count
            self.reductions = reductions
            return self

    def _validate_lr_floor(self, current_lr):
        if current_lr < self.min_lr:
            raise ValueError("optimizer.lr must not be below min_lr")


def _is_better(metric, best, *, mode, threshold, threshold_mode):
    if mode == "min":
        if not metric < best:
            return False
        improvement = Fraction.from_float(best) - Fraction.from_float(metric)
    else:
        if not metric > best:
            return False
        improvement = Fraction.from_float(metric) - Fraction.from_float(best)

    threshold_fraction = Fraction.from_float(threshold)
    if threshold_mode == "rel":
        threshold_fraction *= abs(Fraction.from_float(best))
    return improvement > threshold_fraction


def _reduced_lr(current_lr, factor, min_lr):
    # factor is in (0, 1), so overflow is impossible. Underflow to zero is
    # legitimate arithmetic, but the repository's optimizer LR contract is
    # strictly positive. Clamp a zero floor to the smallest positive binary64.
    with np.errstate(under="ignore"):
        candidate = current_lr * factor
    floor = max(min_lr, _MIN_POSITIVE_LR)
    return max(candidate, floor)


def _assign_optimizer_lr(optimizer, target_lr, *, previous_lr):
    """Set LR and roll back even when a property mutates before raising."""
    if target_lr == previous_lr:
        return target_lr

    try:
        optimizer.lr = target_lr
        observed = _optimizer_lr(optimizer)
        if observed != target_lr:
            raise RuntimeError("optimizer.lr assignment did not preserve requested value")
    except BaseException:
        try:
            optimizer.lr = previous_lr
            restored = _optimizer_lr(optimizer)
            if restored != previous_lr:
                raise RuntimeError("optimizer.lr rollback did not restore previous value")
        except BaseException as rollback_error:
            raise RuntimeError("optimizer lr rollback failed") from rollback_error
        raise
    return target_lr


def _optimizer_lr(optimizer):
    try:
        value = optimizer.lr
    except AttributeError as exc:
        raise TypeError("optimizer must expose an lr attribute") from exc
    return _positive_finite_real("optimizer.lr", value)


def _choice(name, value, choices):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        joined = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {joined}")
    return value


def _integer(name, value, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _nonnegative_integer(name, value):
    return _integer(name, value, minimum=0)


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
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_finite_real(name, value):
    value = _finite_real(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _factor(value, *, name="factor"):
    value = _finite_real(name, value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
