"""Checkpointable early-stopping control for validation metrics."""

from collections.abc import Mapping
from fractions import Fraction
import math
import numbers
import threading

import numpy as np


_FORMAT_VERSION = 1
_TYPE_NAME = "EarlyStopping"


def _normalize_mode(mode):
    if not isinstance(mode, str):
        raise TypeError("mode must be 'min' or 'max'")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
    return mode


def _normalize_threshold_mode(threshold_mode):
    if not isinstance(threshold_mode, str):
        raise TypeError("threshold_mode must be 'abs' or 'rel'")
    if threshold_mode not in {"abs", "rel"}:
        raise ValueError("threshold_mode must be 'abs' or 'rel'")
    return threshold_mode


def _normalize_nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _normalize_finite_real(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must fit in float64") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _normalize_bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _is_improvement(metric, best, mode, min_delta, threshold_mode):
    metric_exact = Fraction.from_float(metric)
    best_exact = Fraction.from_float(best)
    delta = best_exact - metric_exact if mode == "min" else metric_exact - best_exact
    if delta <= 0:
        return False

    threshold = Fraction.from_float(min_delta)
    if threshold_mode == "rel":
        threshold *= abs(best_exact)
    return delta > threshold


class EarlyStopping:
    """Track a metric and signal when consecutive non-improvement exceeds patience.

    ``patience=N`` tolerates exactly ``N`` consecutive non-improving observations
    after a baseline/best value. The next non-improving observation makes the
    stop state sticky. Call :meth:`reset` to begin a fresh monitoring window.
    """

    def __init__(
        self,
        *,
        mode="min",
        patience=0,
        min_delta=0.0,
        threshold_mode="abs",
    ):
        self._mode = _normalize_mode(mode)
        self._patience = _normalize_nonnegative_integer(patience, "patience")
        self._min_delta = _normalize_finite_real(
            min_delta, "min_delta", minimum=0.0
        )
        self._threshold_mode = _normalize_threshold_mode(threshold_mode)
        self._lock = threading.RLock()
        self._reset_runtime()

    def _reset_runtime(self):
        self._best = None
        self._num_bad_epochs = 0
        self._step_count = 0
        self._stopped = False
        self._stopped_step = None

    @property
    def mode(self):
        return self._mode

    @property
    def patience(self):
        return self._patience

    @property
    def min_delta(self):
        return self._min_delta

    @property
    def threshold_mode(self):
        return self._threshold_mode

    @property
    def best(self):
        with self._lock:
            return self._best

    @property
    def num_bad_epochs(self):
        with self._lock:
            return self._num_bad_epochs

    @property
    def step_count(self):
        with self._lock:
            return self._step_count

    @property
    def stopped(self):
        with self._lock:
            return self._stopped

    @property
    def should_stop(self):
        return self.stopped

    @property
    def stopped_step(self):
        with self._lock:
            return self._stopped_step

    def step(self, metric):
        """Observe one metric and return the current sticky stop decision."""
        with self._lock:
            metric = _normalize_finite_real(metric, "metric")
            if self._stopped:
                return True

            next_step = self._step_count + 1
            if self._best is None:
                self._best = metric
                self._num_bad_epochs = 0
                self._step_count = next_step
                return False

            if _is_improvement(
                metric,
                self._best,
                self._mode,
                self._min_delta,
                self._threshold_mode,
            ):
                self._best = metric
                self._num_bad_epochs = 0
                self._step_count = next_step
                return False

            next_bad = self._num_bad_epochs + 1
            self._num_bad_epochs = next_bad
            self._step_count = next_step
            if next_bad > self._patience:
                self._stopped = True
                self._stopped_step = next_step
            return self._stopped

    def reset(self):
        """Reset runtime observations while preserving monitor configuration."""
        with self._lock:
            self._reset_runtime()
            return self

    def state_dict(self):
        """Return a strict-JSON-safe independent monitor state."""
        with self._lock:
            return {
                "format_version": _FORMAT_VERSION,
                "type": _TYPE_NAME,
                "mode": self._mode,
                "patience": self._patience,
                "min_delta": self._min_delta,
                "threshold_mode": self._threshold_mode,
                "best": self._best,
                "num_bad_epochs": self._num_bad_epochs,
                "step_count": self._step_count,
                "stopped": self._stopped,
                "stopped_step": self._stopped_step,
            }

    def load_state_dict(self, state):
        """Validate a complete checkpoint before transactionally restoring it."""
        if not isinstance(state, Mapping):
            raise TypeError("early stopping state must be a mapping")

        required = {
            "format_version",
            "type",
            "mode",
            "patience",
            "min_delta",
            "threshold_mode",
            "best",
            "num_bad_epochs",
            "step_count",
            "stopped",
            "stopped_step",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                "early stopping state is missing required keys: "
                + ", ".join(sorted(missing))
            )

        version = _normalize_nonnegative_integer(
            state["format_version"], "format_version"
        )
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported early stopping format_version: {version}")
        if state["type"] != _TYPE_NAME:
            raise ValueError(f"early stopping state type must be {_TYPE_NAME!r}")

        mode = _normalize_mode(state["mode"])
        patience = _normalize_nonnegative_integer(state["patience"], "patience")
        min_delta = _normalize_finite_real(
            state["min_delta"], "min_delta", minimum=0.0
        )
        threshold_mode = _normalize_threshold_mode(state["threshold_mode"])
        step_count = _normalize_nonnegative_integer(state["step_count"], "step_count")
        num_bad_epochs = _normalize_nonnegative_integer(
            state["num_bad_epochs"], "num_bad_epochs"
        )
        stopped = _normalize_bool(state["stopped"], "stopped")

        raw_best = state["best"]
        best = None if raw_best is None else _normalize_finite_real(raw_best, "best")
        raw_stopped_step = state["stopped_step"]
        stopped_step = (
            None
            if raw_stopped_step is None
            else _normalize_nonnegative_integer(raw_stopped_step, "stopped_step")
        )

        if step_count == 0:
            if best is not None:
                raise ValueError("empty early stopping state must have best=None")
            if num_bad_epochs != 0:
                raise ValueError("empty early stopping state must have num_bad_epochs=0")
            if stopped or stopped_step is not None:
                raise ValueError("empty early stopping state cannot be stopped")
        else:
            if best is None:
                raise ValueError("non-empty early stopping state must have a best metric")
            if stopped:
                if num_bad_epochs != patience + 1:
                    raise ValueError(
                        "stopped early stopping state must have num_bad_epochs=patience+1"
                    )
                if stopped_step != step_count:
                    raise ValueError(
                        "stopped early stopping state must stop at the current step_count"
                    )
                if step_count < patience + 2:
                    raise ValueError("stopped early stopping state has impossible step_count")
            else:
                if num_bad_epochs > patience:
                    raise ValueError(
                        "active early stopping state cannot exceed configured patience"
                    )
                if stopped_step is not None:
                    raise ValueError(
                        "active early stopping state must have stopped_step=None"
                    )

        with self._lock:
            self._mode = mode
            self._patience = patience
            self._min_delta = min_delta
            self._threshold_mode = threshold_mode
            self._best = best
            self._num_bad_epochs = num_bad_epochs
            self._step_count = step_count
            self._stopped = stopped
            self._stopped_step = stopped_step
            return self
