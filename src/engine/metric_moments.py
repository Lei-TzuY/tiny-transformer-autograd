"""Overflow-resistant weighted streaming mean/variance statistics."""

from collections.abc import Mapping
from numbers import Integral, Real
import math
import threading

import numpy as np


_STATE_VERSION = 1
_STATE_TYPE = "WeightedStreamingMoments"
_MIN_M2_EXPONENT = -4096
_MAX_M2_EXPONENT = 4096


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_real(name, value):
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _nonnegative_int(name, value):
    normalized = _integer(name, value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _m2_exponent(value):
    exponent = _integer("streaming moments m2_exponent", value)
    if not (_MIN_M2_EXPONENT <= exponent <= _MAX_M2_EXPONENT):
        raise ValueError(
            "streaming moments m2_exponent is outside the supported range"
        )
    return exponent


class _ScaledNonnegative:
    """Non-negative binary floating value with a separately stored exponent."""

    __slots__ = ("mantissa", "exponent")

    def __init__(self, mantissa=0.0, exponent=0):
        if mantissa == 0.0:
            self.mantissa = 0.0
            self.exponent = 0
            return
        normalized, shift = math.frexp(float(mantissa))
        self.mantissa = normalized
        self.exponent = int(exponent) + shift

    @classmethod
    def from_float(cls, value):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("scaled value must be finite and non-negative")
        if value == 0.0:
            return cls()
        mantissa, exponent = math.frexp(value)
        return cls(mantissa, exponent)

    @property
    def is_zero(self):
        return self.mantissa == 0.0

    def add(self, other):
        if self.is_zero:
            return _ScaledNonnegative(other.mantissa, other.exponent)
        if other.is_zero:
            return _ScaledNonnegative(self.mantissa, self.exponent)
        left, right = self, other
        if left.exponent < right.exponent:
            left, right = right, left
        gap = right.exponent - left.exponent
        try:
            aligned = math.ldexp(right.mantissa, gap)
        except OverflowError:  # defensive: gap is non-positive by construction
            aligned = 0.0
        return _ScaledNonnegative(left.mantissa + aligned, left.exponent)

    def multiply(self, other):
        if self.is_zero or other.is_zero:
            return _ScaledNonnegative()
        return _ScaledNonnegative(
            self.mantissa * other.mantissa,
            self.exponent + other.exponent,
        )

    def divide_float(self, divisor):
        if divisor <= 0.0 or not math.isfinite(divisor):
            raise ValueError("scaled divisor must be finite and positive")
        if self.is_zero:
            return _ScaledNonnegative()
        divisor_mantissa, divisor_exponent = math.frexp(divisor)
        return _ScaledNonnegative(
            self.mantissa / divisor_mantissa,
            self.exponent - divisor_exponent,
        )

    def sqrt(self):
        if self.is_zero:
            return _ScaledNonnegative()
        mantissa = self.mantissa
        exponent = self.exponent
        if exponent % 2:
            mantissa *= 2.0
            exponent -= 1
        return _ScaledNonnegative(math.sqrt(mantissa), exponent // 2)

    def to_float(self):
        if self.is_zero:
            return 0.0, False, False
        try:
            value = math.ldexp(self.mantissa, self.exponent)
        except OverflowError:
            return None, True, False
        if not math.isfinite(value):
            return None, True, False
        underflow = value == 0.0
        return float(value), False, underflow


def _scaled_abs_difference(left, right):
    if left == right:
        return _ScaledNonnegative()
    if (left < 0.0 < right) or (right < 0.0 < left):
        return _ScaledNonnegative.from_float(abs(left)).add(
            _ScaledNonnegative.from_float(abs(right))
        )
    return _ScaledNonnegative.from_float(abs(left - right))


def _combine_mean(left_mean, left_weight, right_mean, right_weight, total_weight):
    right_fraction = right_weight / total_weight
    same_sign = (
        (left_mean >= 0.0 and right_mean >= 0.0)
        or (left_mean <= 0.0 and right_mean <= 0.0)
    )
    if same_sign:
        candidate = left_mean + right_fraction * (right_mean - left_mean)
    else:
        left_fraction = left_weight / total_weight
        candidate = left_mean * left_fraction + right_mean * right_fraction
    if not math.isfinite(candidate):
        raise ValueError("weighted metric mean is not representable as float64")
    return float(candidate)


def _cross_weight(left_weight, right_weight, total_weight):
    return (
        _ScaledNonnegative.from_float(left_weight)
        .multiply(_ScaledNonnegative.from_float(right_weight))
        .divide_float(total_weight)
    )


def _merge_state(left_mean, left_weight, left_m2, right_mean, right_weight, right_m2):
    total_weight = left_weight + right_weight
    if not math.isfinite(total_weight):
        raise ValueError("total metric weight must remain finite")
    mean = _combine_mean(
        left_mean,
        left_weight,
        right_mean,
        right_weight,
        total_weight,
    )
    delta = _scaled_abs_difference(left_mean, right_mean)
    cross = delta.multiply(delta).multiply(
        _cross_weight(left_weight, right_weight, total_weight)
    )
    m2 = left_m2.add(right_m2).add(cross)
    return mean, float(total_weight), m2


def _variance_snapshot(m2, total_weight):
    if total_weight == 0.0:
        return None, False, False, None, False, False
    variance_scaled = m2.divide_float(total_weight)
    variance, variance_overflow, variance_underflow = variance_scaled.to_float()
    std, std_overflow, std_underflow = variance_scaled.sqrt().to_float()
    return (
        variance,
        variance_overflow,
        variance_underflow,
        std,
        std_overflow,
        std_underflow,
    )


class WeightedStreamingMoments:
    """Track weighted population mean, variance, and standard deviation online."""

    def __init__(self):
        self._mean = None
        self._total_weight = 0.0
        self._observation_count = 0
        self._m2 = _ScaledNonnegative()
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

    @property
    def variance(self):
        with self._lock:
            return _variance_snapshot(self._m2, self._total_weight)[0]

    @property
    def std(self):
        with self._lock:
            return _variance_snapshot(self._m2, self._total_weight)[3]

    def statistics(self):
        """Return one strict-JSON-safe, internally consistent statistics snapshot."""
        with self._lock:
            (
                variance,
                variance_overflow,
                variance_underflow,
                std,
                std_overflow,
                std_underflow,
            ) = _variance_snapshot(self._m2, self._total_weight)
            return {
                "mean": self._mean,
                "variance": variance,
                "variance_overflow": variance_overflow,
                "variance_underflow": variance_underflow,
                "std": std,
                "std_overflow": std_overflow,
                "std_underflow": std_underflow,
                "total_weight": self._total_weight,
                "observation_count": self._observation_count,
            }

    def update(self, value, *, weight=1.0):
        """Add one finite scalar observation and return a statistics snapshot."""
        value = _finite_real("value", value)
        weight = _positive_real("weight", weight)
        with self._lock:
            if self._observation_count == 0:
                self._mean = value
                self._total_weight = weight
                self._observation_count = 1
                self._m2 = _ScaledNonnegative()
                return self.statistics()

            mean, total_weight, m2 = _merge_state(
                self._mean,
                self._total_weight,
                self._m2,
                value,
                weight,
                _ScaledNonnegative(),
            )
            self._mean = mean
            self._total_weight = total_weight
            self._observation_count += 1
            self._m2 = m2
            return self.statistics()

    def merge(self, other):
        """Atomically merge another moments accumulator into this one."""
        if not isinstance(other, WeightedStreamingMoments):
            raise TypeError("other must be a WeightedStreamingMoments")

        if other is self:
            with self._lock:
                if self._observation_count == 0:
                    return self.statistics()
                mean, total_weight, m2 = _merge_state(
                    self._mean,
                    self._total_weight,
                    self._m2,
                    self._mean,
                    self._total_weight,
                    self._m2,
                )
                self._mean = mean
                self._total_weight = total_weight
                self._observation_count *= 2
                self._m2 = m2
                return self.statistics()

        first, second = (self, other) if id(self) < id(other) else (other, self)
        with first._lock:
            with second._lock:
                if other._observation_count == 0:
                    return self.statistics()
                if self._observation_count == 0:
                    self._mean = other._mean
                    self._total_weight = other._total_weight
                    self._observation_count = other._observation_count
                    self._m2 = _ScaledNonnegative(
                        other._m2.mantissa,
                        other._m2.exponent,
                    )
                    return self.statistics()

                mean, total_weight, m2 = _merge_state(
                    self._mean,
                    self._total_weight,
                    self._m2,
                    other._mean,
                    other._total_weight,
                    other._m2,
                )
                self._mean = mean
                self._total_weight = total_weight
                self._observation_count += other._observation_count
                self._m2 = m2
                return self.statistics()

    def reset(self):
        with self._lock:
            self._mean = None
            self._total_weight = 0.0
            self._observation_count = 0
            self._m2 = _ScaledNonnegative()

    def state_dict(self):
        """Return a strict-JSON-safe checkpoint state."""
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "mean": self._mean,
                "total_weight": self._total_weight,
                "observation_count": self._observation_count,
                "m2_mantissa": self._m2.mantissa,
                "m2_exponent": self._m2.exponent,
            }

    def load_state_dict(self, state):
        """Validate and transactionally restore a checkpoint state."""
        if not isinstance(state, Mapping):
            raise TypeError("streaming moments state must be a mapping")

        version = _nonnegative_int("streaming moments version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported streaming moments version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"streaming moments type must be {_STATE_TYPE!r}")

        count = _nonnegative_int(
            "streaming moments observation_count",
            state.get("observation_count"),
        )
        total_weight = _finite_real(
            "streaming moments total_weight",
            state.get("total_weight"),
        )
        if total_weight < 0.0:
            raise ValueError("streaming moments total_weight must be non-negative")

        mantissa = _finite_real(
            "streaming moments m2_mantissa",
            state.get("m2_mantissa"),
        )
        exponent = _m2_exponent(state.get("m2_exponent"))
        if mantissa < 0.0:
            raise ValueError("streaming moments m2_mantissa must be non-negative")
        if mantissa == 0.0:
            if exponent != 0:
                raise ValueError("zero streaming moments M2 must use exponent 0")
            candidate_m2 = _ScaledNonnegative()
        else:
            if not (0.5 <= mantissa < 1.0):
                raise ValueError("streaming moments m2_mantissa must be normalized")
            candidate_m2 = _ScaledNonnegative(mantissa, exponent)

        mean_value = state.get("mean")
        if count == 0:
            if total_weight != 0.0:
                raise ValueError("empty streaming moments state must have zero total_weight")
            if mean_value is not None:
                raise ValueError("empty streaming moments state must have mean=None")
            if not candidate_m2.is_zero:
                raise ValueError("empty streaming moments state must have zero M2")
            candidate_mean = None
        else:
            if total_weight <= 0.0:
                raise ValueError("non-empty streaming moments state must have positive total_weight")
            candidate_mean = _finite_real("streaming moments mean", mean_value)
            if count == 1 and not candidate_m2.is_zero:
                raise ValueError("single-observation streaming moments state must have zero M2")

        with self._lock:
            self._mean = candidate_mean
            self._total_weight = total_weight
            self._observation_count = count
            self._m2 = candidate_m2
