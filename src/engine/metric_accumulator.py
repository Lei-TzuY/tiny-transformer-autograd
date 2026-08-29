"""Overflow-resistant weighted scalar metric accumulation."""

from collections.abc import Mapping
from numbers import Integral, Real
import threading

import numpy as np


_STATE_VERSION = 1
_STATE_TYPE = "WeightedMetricAccumulator"


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_real(name, value):
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _combine_means(left_mean, left_weight, right_mean, right_weight):
    """Combine two non-empty weighted means without raw weighted sums."""
    total_weight = left_weight + right_weight
    if not np.isfinite(total_weight):
        raise ValueError("total metric weight must remain finite")

    right_fraction = right_weight / total_weight
    same_sign = (
        (left_mean >= 0.0 and right_mean >= 0.0)
        or (left_mean <= 0.0 and right_mean <= 0.0)
    )

    if same_sign:
        # The subtraction of same-sign finite endpoints is representable.
        # Interpolating from one endpoint also avoids adding two large
        # same-sign weighted terms that may round through overflow.
        candidate = left_mean + right_fraction * (right_mean - left_mean)
    else:
        # Opposite-sign subtraction can overflow even when the convex result is
        # finite. Scale both endpoints first; the resulting terms have opposite
        # signs and each is bounded by its original endpoint magnitude.
        left_fraction = left_weight / total_weight
        candidate = left_mean * left_fraction + right_mean * right_fraction

    if not np.isfinite(candidate):
        raise ValueError("weighted metric mean is not representable as float64")
    return float(candidate), float(total_weight)


class WeightedMetricAccumulator:
    """Track a checkpointable weighted mean of finite scalar observations.

    The implementation stores the current mean and total weight instead of a
    raw weighted sum. This avoids intermediates such as ``value * weight`` that
    can overflow even when the final weighted mean is finite.
    """

    def __init__(self):
        self._mean = None
        self._total_weight = 0.0
        self._observation_count = 0
        self._lock = threading.RLock()

    @property
    def mean(self):
        with self._lock:
            return self._mean

    @property
    def total_weight(self):
        with self._lock:
            return self._total_weight

    @property
    def observation_count(self):
        with self._lock:
            return self._observation_count

    def update(self, value, *, weight=1.0):
        """Add one finite scalar observation and return the new weighted mean."""
        value = _finite_real("value", value)
        weight = _positive_real("weight", weight)

        with self._lock:
            if self._observation_count == 0:
                self._mean = value
                self._total_weight = weight
                self._observation_count = 1
                return self._mean

            candidate_mean, candidate_weight = _combine_means(
                self._mean,
                self._total_weight,
                value,
                weight,
            )
            self._mean = candidate_mean
            self._total_weight = candidate_weight
            self._observation_count += 1
            return self._mean

    def merge(self, other):
        """Merge another accumulator snapshot into this instance."""
        if not isinstance(other, WeightedMetricAccumulator):
            raise TypeError("other must be a WeightedMetricAccumulator")

        other_state = other.state_dict()
        other_count = other_state["observation_count"]
        if other_count == 0:
            with self._lock:
                return self._mean

        with self._lock:
            if self._observation_count == 0:
                self._mean = other_state["mean"]
                self._total_weight = other_state["total_weight"]
                self._observation_count = other_count
                return self._mean

            candidate_mean, candidate_weight = _combine_means(
                self._mean,
                self._total_weight,
                other_state["mean"],
                other_state["total_weight"],
            )
            candidate_count = self._observation_count + other_count

            self._mean = candidate_mean
            self._total_weight = candidate_weight
            self._observation_count = candidate_count
            return self._mean

    def reset(self):
        """Discard all accumulated observations."""
        with self._lock:
            self._mean = None
            self._total_weight = 0.0
            self._observation_count = 0

    def state_dict(self):
        """Return a strict-JSON-safe independent state mapping."""
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "mean": self._mean,
                "total_weight": self._total_weight,
                "observation_count": self._observation_count,
            }

    def load_state_dict(self, state):
        """Validate and transactionally restore accumulator state."""
        if not isinstance(state, Mapping):
            raise TypeError("metric accumulator state must be a mapping")

        version = _nonnegative_int("metric accumulator version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported metric accumulator version: {version}")

        state_type = state.get("type")
        if state_type != _STATE_TYPE:
            raise ValueError(f"metric accumulator type must be {_STATE_TYPE!r}")

        observation_count = _nonnegative_int(
            "metric accumulator observation_count",
            state.get("observation_count"),
        )
        total_weight = _finite_real(
            "metric accumulator total_weight",
            state.get("total_weight"),
        )
        if total_weight < 0.0:
            raise ValueError("metric accumulator total_weight must be non-negative")

        mean_value = state.get("mean")
        if observation_count == 0:
            if total_weight != 0.0:
                raise ValueError(
                    "empty metric accumulator state must have zero total_weight"
                )
            if mean_value is not None:
                raise ValueError("empty metric accumulator state must have mean=None")
            candidate_mean = None
        else:
            if total_weight <= 0.0:
                raise ValueError(
                    "non-empty metric accumulator state must have positive total_weight"
                )
            candidate_mean = _finite_real("metric accumulator mean", mean_value)

        with self._lock:
            self._mean = candidate_mean
            self._total_weight = total_weight
            self._observation_count = observation_count
