"""Weighted empirical diagonal-Fisher diagnostics for tiny-transformer.

Each capture treats the currently bound gradients as one empirical-Fisher
observation and accumulates a weighted mean of elementwise squared gradients.
Second moments are stored as a root scale plus a normalized diagonal buffer so
finite float64 gradients never need to be squared at their original magnitude.
"""

from collections.abc import Iterable, Mapping
from numbers import Integral, Real
import sys
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "DiagonalFisherEstimator"
_MAX_COUNT = sys.maxsize


def _positive_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_int(name, value, *, maximum=_MAX_COUNT):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        materialized = (parameters,)
    else:
        try:
            materialized = tuple(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be a Tensor or iterable of Tensors") from exc

    seen = set()
    for index, parameter in enumerate(materialized):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(marker)
        if not isinstance(parameter.requires_grad, (bool, np.bool_)):
            raise TypeError(f"parameter {index} requires_grad must be boolean")
        if not bool(parameter.requires_grad):
            raise ValueError(f"parameter {index} must require gradients")
    return materialized


def _independent_array(value):
    """Return an ordinary float64 ndarray, preserving scalar shape ``()``."""
    return np.array(value, dtype=np.float64, copy=True, subok=False)


def _float64_array_copy(array, *, name, shape):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    if array.dtype.itemsize > np.dtype(np.float64).itemsize:
        limit = np.array(np.finfo(np.float64).max, dtype=array.dtype)
        with np.errstate(over="ignore", invalid="raise", under="ignore"):
            if np.any(np.abs(array) > limit):
                raise ValueError(f"{name} must fit float64")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = _independent_array(np.asarray(array, dtype=np.float64))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must fit float64")
    return result


def _max_abs(array):
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _canonicalize(scale, diagonal):
    diagonal = _independent_array(diagonal)
    if diagonal.size == 0:
        return 0.0, np.zeros_like(diagonal)
    peak = float(np.max(diagonal))
    if peak == 0.0:
        return 0.0, np.zeros_like(diagonal)
    if not np.isfinite(peak) or peak < 0.0:
        raise ValueError("diagonal Fisher state is not finite and non-negative")

    root = float(np.sqrt(peak))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        candidate_scale = scale * root
        normalized = _independent_array(diagonal / peak)
    if not np.isfinite(candidate_scale):
        raise ValueError("diagonal Fisher root scale is not representable")
    if candidate_scale == 0.0 and scale > 0.0:
        # The physical Fisher entry is smaller than the least representable root
        # scale. Keep the old scale plus the tiny normalized diagonal instead of
        # erasing a positive value through canonicalization underflow.
        return scale, diagonal
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("diagonal Fisher normalized state is invalid")
    return candidate_scale, normalized


def _combine_states(left, left_weight, right, right_weight):
    if left_weight == 0.0:
        return {
            "scale": right["scale"],
            "diagonal": _independent_array(right["diagonal"]),
        }
    if right_weight == 0.0:
        return {
            "scale": left["scale"],
            "diagonal": _independent_array(left["diagonal"]),
        }

    total = left_weight + right_weight
    if not np.isfinite(total):
        raise OverflowError("diagonal Fisher total weight overflow")
    common = max(left["scale"], right["scale"])
    if common == 0.0:
        return {"scale": 0.0, "diagonal": np.zeros_like(left["diagonal"])}

    left_ratio = 0.0 if left["scale"] == 0.0 else left["scale"] / common
    right_ratio = 0.0 if right["scale"] == 0.0 else right["scale"] / common
    left_fraction = left_weight / total
    right_fraction = right_weight / total
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        combined = (
            left_fraction * left["diagonal"] * (left_ratio * left_ratio)
            + right_fraction * right["diagonal"] * (right_ratio * right_ratio)
        )
    scale, combined = _canonicalize(common, combined)
    return {"scale": scale, "diagonal": combined}


def _state_from_gradient(gradient):
    scale = _max_abs(gradient)
    if scale == 0.0:
        return {"scale": 0.0, "diagonal": np.zeros_like(gradient)}
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        normalized = gradient / scale
        diagonal = normalized * normalized
    scale, diagonal = _canonicalize(scale, diagonal)
    return {"scale": scale, "diagonal": diagonal}


def _descriptor_normalize(mantissa, exponent):
    if mantissa == 0.0:
        return 0.0, 0
    fraction, shift = np.frexp(mantissa)
    return float(fraction), int(exponent + shift)


def _descriptor_from_state(state):
    scale = state["scale"]
    diagonal = state["diagonal"]
    if scale == 0.0 or diagonal.size == 0:
        return 0.0, 0
    diagonal_sum = float(np.sum(diagonal, dtype=np.float64))
    if diagonal_sum == 0.0:
        return 0.0, 0
    scale_mantissa, scale_exponent = np.frexp(scale)
    mantissa = float(scale_mantissa * scale_mantissa * diagonal_sum)
    return _descriptor_normalize(mantissa, 2 * int(scale_exponent))


def _descriptor_add(left, right):
    if left[0] == 0.0:
        return right
    if right[0] == 0.0:
        return left
    if left[1] < right[1]:
        left, right = right, left
    shift = right[1] - left[1]
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        mantissa = left[0] + float(np.ldexp(right[0], shift))
    return _descriptor_normalize(mantissa, left[1])


def _descriptor_to_float(descriptor):
    mantissa, exponent = descriptor
    if mantissa == 0.0:
        return 0.0, False, False
    max_value = np.finfo(np.float64).max
    max_mantissa, max_exponent = np.frexp(max_value)
    if exponent > max_exponent or (
        exponent == max_exponent and mantissa > max_mantissa
    ):
        return None, True, False
    with np.errstate(over="ignore", invalid="raise", under="ignore"):
        value = float(np.ldexp(mantissa, exponent))
    if not np.isfinite(value):
        return None, True, False
    underflow = value == 0.0
    return value, False, underflow


def _normalize_loaded_state(raw, shape, index):
    if not isinstance(raw, Mapping):
        raise TypeError(f"diagonal Fisher state[{index}] must be a mapping")
    scale_value = raw.get("scale")
    if isinstance(scale_value, (bool, np.bool_)) or not isinstance(scale_value, Real):
        raise TypeError(f"diagonal Fisher state[{index}] scale must be a real number")
    try:
        scale = float(scale_value)
    except OverflowError as exc:
        raise ValueError(f"diagonal Fisher state[{index}] scale must fit float64") from exc
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"diagonal Fisher state[{index}] scale must be finite and non-negative")

    diagonal = _float64_array_copy(
        raw.get("diagonal"),
        name=f"diagonal Fisher state[{index}] diagonal",
        shape=shape,
    )
    if np.any(diagonal < 0.0):
        raise ValueError(f"diagonal Fisher state[{index}] diagonal must be non-negative")
    if scale == 0.0:
        if np.any(diagonal != 0.0):
            raise ValueError(f"zero-scale diagonal Fisher state[{index}] must be zero")
        return {"scale": 0.0, "diagonal": np.zeros(shape, dtype=np.float64)}
    if diagonal.size == 0 or not np.any(diagonal > 0.0):
        raise ValueError(f"positive-scale diagonal Fisher state[{index}] must be nonzero")
    scale, diagonal = _canonicalize(scale, diagonal)
    return {"scale": scale, "diagonal": diagonal}


class DiagonalFisherEstimator:
    """Accumulate a weighted empirical diagonal Fisher from live gradients."""

    def __init__(self, parameters):
        self.parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self.parameters)
        self._states = tuple(
            {"scale": 0.0, "diagonal": np.zeros(shape, dtype=np.float64)}
            for shape in self._shapes
        )
        self._total_weight = 0.0
        self._observation_count = 0
        self._lock = threading.RLock()

    @property
    def observation_count(self):
        with self._lock:
            return self._observation_count

    @property
    def total_weight(self):
        with self._lock:
            return self._total_weight

    def _validate_live_parameters(self):
        gradients = []
        for index, (parameter, shape) in enumerate(zip(self.parameters, self._shapes)):
            if parameter.shape != shape:
                raise ValueError(
                    f"parameter {index} shape changed from {shape} to {parameter.shape}"
                )
            if not isinstance(parameter.requires_grad, (bool, np.bool_)):
                raise TypeError(f"parameter {index} requires_grad must be boolean")
            if not bool(parameter.requires_grad):
                raise ValueError(f"parameter {index} no longer requires gradients")
            gradient = parameter.grad
            if gradient is None:
                gradients.append(np.zeros(shape, dtype=np.float64))
                continue
            gradients.append(
                _float64_array_copy(
                    gradient,
                    name=f"gradient for parameter {index}",
                    shape=shape,
                )
            )
        return tuple(gradients)

    def capture(self, *, weight=1.0):
        """Capture one weighted empirical-Fisher observation from live gradients."""
        weight = _positive_real("weight", weight)
        with self._lock:
            if self._observation_count >= _MAX_COUNT:
                raise OverflowError("diagonal Fisher observation count reached its maximum")
            total_weight = self._total_weight + weight
            if not np.isfinite(total_weight):
                raise OverflowError("diagonal Fisher total weight overflow")
            gradients = self._validate_live_parameters()

            candidate_states = []
            for state, gradient in zip(self._states, gradients):
                sample = _state_from_gradient(gradient)
                candidate_states.append(
                    _combine_states(state, self._total_weight, sample, weight)
                )

            self._states = tuple(candidate_states)
            self._total_weight = total_weight
            self._observation_count += 1
        return self

    def scaled_diagonals(self):
        """Return independent ``scale**2 * diagonal`` representations."""
        with self._lock:
            if self._observation_count == 0:
                raise RuntimeError("diagonal Fisher has no observations")
            return tuple(
                {
                    "scale": state["scale"],
                    "diagonal": _independent_array(state["diagonal"]),
                }
                for state in self._states
            )

    def diagonals(self):
        """Return ordinary float64 diagonal arrays when every value fits."""
        with self._lock:
            if self._observation_count == 0:
                raise RuntimeError("diagonal Fisher has no observations")
            result = []
            sqrt_max = float(np.sqrt(np.finfo(np.float64).max))
            for index, state in enumerate(self._states):
                scale = state["scale"]
                if scale > sqrt_max and np.any(state["diagonal"] > 0.0):
                    raise OverflowError(
                        f"diagonal Fisher parameter {index} is not representable in float64"
                    )
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    values = (scale * scale) * state["diagonal"]
                if not np.all(np.isfinite(values)):
                    raise OverflowError(
                        f"diagonal Fisher parameter {index} is not representable in float64"
                    )
                result.append(_independent_array(values))
            return tuple(result)

    def trace_report(self):
        """Return a strict-JSON-safe report of the total diagonal-Fisher trace."""
        with self._lock:
            if self._observation_count == 0:
                return {
                    "trace": None,
                    "trace_overflow": False,
                    "trace_underflow": False,
                    "total_weight": 0.0,
                    "observation_count": 0,
                    "parameter_count": len(self.parameters),
                    "reason": "no_observations",
                }
            descriptor = (0.0, 0)
            for state in self._states:
                descriptor = _descriptor_add(descriptor, _descriptor_from_state(state))
            trace, overflow, underflow = _descriptor_to_float(descriptor)
            return {
                "trace": trace,
                "trace_overflow": overflow,
                "trace_underflow": underflow,
                "total_weight": self._total_weight,
                "observation_count": self._observation_count,
                "parameter_count": len(self.parameters),
                "reason": "overflow" if overflow else "ok",
            }

    def merge(self, other):
        """Merge another compatible estimator into this estimator."""
        if not isinstance(other, DiagonalFisherEstimator):
            raise TypeError("other must be a DiagonalFisherEstimator")
        if self is other:
            locks = (self._lock,)
        else:
            locks = tuple(
                estimator._lock for estimator in sorted((self, other), key=id)
            )
        for lock in locks:
            lock.acquire()
        try:
            if self._shapes != other._shapes:
                raise ValueError("diagonal Fisher estimator parameter shapes do not match")
            if other._observation_count == 0:
                return self
            if self._observation_count > _MAX_COUNT - other._observation_count:
                raise OverflowError("diagonal Fisher observation count overflow")
            total_weight = self._total_weight + other._total_weight
            if not np.isfinite(total_weight):
                raise OverflowError("diagonal Fisher total weight overflow")
            source_states = tuple(
                {
                    "scale": state["scale"],
                    "diagonal": _independent_array(state["diagonal"]),
                }
                for state in other._states
            )
            source_weight = other._total_weight
            source_count = other._observation_count
            candidate_states = tuple(
                _combine_states(left, self._total_weight, right, source_weight)
                for left, right in zip(self._states, source_states)
            )
            self._states = candidate_states
            self._total_weight = total_weight
            self._observation_count += source_count
            return self
        finally:
            for lock in reversed(locks):
                lock.release()

    def reset(self):
        with self._lock:
            self._states = tuple(
                {"scale": 0.0, "diagonal": np.zeros(shape, dtype=np.float64)}
                for shape in self._shapes
            )
            self._total_weight = 0.0
            self._observation_count = 0
        return self

    def state_dict(self):
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "total_weight": self._total_weight,
                "observation_count": self._observation_count,
                "states": [
                    {
                        "scale": state["scale"],
                        "diagonal": _independent_array(state["diagonal"]),
                    }
                    for state in self._states
                ],
            }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("diagonal Fisher state must be a mapping")
        version = _nonnegative_int("diagonal Fisher version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported diagonal Fisher version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"diagonal Fisher type must be {_STATE_TYPE!r}")

        count = _nonnegative_int(
            "diagonal Fisher observation_count", state.get("observation_count")
        )
        weight_value = state.get("total_weight")
        if isinstance(weight_value, (bool, np.bool_)) or not isinstance(weight_value, Real):
            raise TypeError("diagonal Fisher total_weight must be a real number")
        try:
            total_weight = float(weight_value)
        except OverflowError as exc:
            raise ValueError("diagonal Fisher total_weight must fit float64") from exc
        if not np.isfinite(total_weight) or total_weight < 0.0:
            raise ValueError("diagonal Fisher total_weight must be finite and non-negative")
        if (count == 0) != (total_weight == 0.0):
            raise ValueError("diagonal Fisher count/weight empty-state invariant is invalid")

        raw_states = state.get("states")
        if isinstance(raw_states, (str, bytes)) or not isinstance(raw_states, Iterable):
            raise TypeError("diagonal Fisher states must be an iterable of mappings")
        raw_states = tuple(raw_states)
        if len(raw_states) != len(self._shapes):
            raise ValueError(
                "diagonal Fisher state count mismatch: "
                f"expected {len(self._shapes)}, got {len(raw_states)}"
            )
        normalized = tuple(
            _normalize_loaded_state(raw, shape, index)
            for index, (raw, shape) in enumerate(zip(raw_states, self._shapes))
        )
        if count == 0 and any(item["scale"] != 0.0 for item in normalized):
            raise ValueError("empty diagonal Fisher state must contain zero diagonals")

        with self._lock:
            for index, (parameter, shape) in enumerate(zip(self.parameters, self._shapes)):
                if parameter.shape != shape:
                    raise ValueError(
                        f"parameter {index} shape changed from {shape} to {parameter.shape}"
                    )
            self._states = normalized
            self._total_weight = total_weight
            self._observation_count = count
        return self
