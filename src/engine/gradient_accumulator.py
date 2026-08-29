"""Reusable weighted gradient accumulation for custom training loops."""

from collections.abc import Iterable, Mapping
from numbers import Real
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "GradientAccumulator"


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        values = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("parameters must be a Tensor or iterable of Tensors")
        values = tuple(parameters)
    seen = set()
    for parameter in values:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
        if id(parameter) in seen:
            raise ValueError("parameters must not contain duplicate Tensors")
        seen.add(id(parameter))
        if not parameter.requires_grad:
            raise ValueError("all parameters must require gradients")
    return values


def _gradient_snapshot(parameter, index):
    gradient = parameter.grad
    if gradient is None:
        return np.zeros(parameter.shape, dtype=np.float64)
    if not isinstance(gradient, np.ndarray):
        raise TypeError(f"gradient {index} must be a NumPy array or None")
    if gradient.shape != parameter.shape:
        raise ValueError(
            f"gradient {index} shape mismatch: expected {parameter.shape}, got {gradient.shape}"
        )
    if not np.issubdtype(gradient.dtype, np.floating):
        raise TypeError(f"gradient {index} must contain floating-point values")
    if not np.isfinite(gradient).all():
        raise ValueError(f"gradient {index} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = np.asarray(gradient, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"gradient {index} must fit in float64")
    return np.array(converted, dtype=np.float64, copy=True)


def _weighted_average(previous, current, previous_weight, weight, total_weight):
    if previous_weight == 0.0:
        return np.array(current, dtype=np.float64, copy=True)
    left_weight = previous_weight / total_weight
    right_weight = weight / total_weight
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        candidate = previous * left_weight + current * right_weight
    if not np.isfinite(candidate).all():
        raise ValueError("accumulated gradients must remain finite")
    return np.array(candidate, dtype=np.float64, copy=True)


class GradientAccumulator:
    """Maintain a transactional online weighted average of parameter gradients.

    ``grad is None`` contributes an exact zero for that parameter. Positive
    ``weight`` values can therefore represent micro-batch sizes or scored-token
    counts while preserving the mathematical weighted mean directly, without
    first materializing an overflow-prone weighted sum.
    """

    def __init__(self, parameters):
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        self._averages = tuple(
            np.zeros(shape, dtype=np.float64) for shape in self._shapes
        )
        self._total_weight = 0.0
        self._accumulation_count = 0
        self._lock = threading.RLock()

    @property
    def parameters(self):
        return self._parameters

    @property
    def total_weight(self):
        with self._lock:
            return self._total_weight

    @property
    def accumulation_count(self):
        with self._lock:
            return self._accumulation_count

    def _validate_binding(self):
        for index, (parameter, shape) in enumerate(zip(self._parameters, self._shapes)):
            if parameter.shape != shape:
                raise ValueError(
                    f"parameter {index} shape changed: expected {shape}, got {parameter.shape}"
                )
            if not parameter.requires_grad:
                raise ValueError(f"parameter {index} no longer requires gradients")

    def accumulate(self, weight=1.0):
        """Incorporate the current ``.grad`` buffers with a positive weight."""
        weight = _positive_real(weight, "weight")
        with self._lock:
            self._validate_binding()
            snapshots = tuple(
                _gradient_snapshot(parameter, index)
                for index, parameter in enumerate(self._parameters)
            )
            total_weight = self._total_weight + weight
            if not np.isfinite(total_weight):
                raise ValueError("total accumulated weight must remain finite")
            candidates = tuple(
                _weighted_average(
                    previous,
                    current,
                    self._total_weight,
                    weight,
                    total_weight,
                )
                for previous, current in zip(self._averages, snapshots)
            )
            self._averages = candidates
            self._total_weight = total_weight
            self._accumulation_count += 1
            return total_weight

    def average_gradients(self):
        """Return independent copies of the current weighted-average buffers."""
        with self._lock:
            if self._accumulation_count == 0:
                raise RuntimeError("no gradients have been accumulated")
            return tuple(np.array(value, copy=True) for value in self._averages)

    def copy_to_grads(self):
        """Install independent float64 copies into the bound Tensor ``.grad`` slots."""
        with self._lock:
            if self._accumulation_count == 0:
                raise RuntimeError("no gradients have been accumulated")
            self._validate_binding()
            candidates = tuple(np.array(value, copy=True) for value in self._averages)
            previous = tuple(parameter.grad for parameter in self._parameters)
            try:
                for parameter, candidate in zip(self._parameters, candidates):
                    parameter.grad = candidate
            except BaseException as original:
                rollback_error = None
                for parameter, old in zip(self._parameters, previous):
                    try:
                        parameter.grad = old
                    except BaseException as exc:
                        if rollback_error is None:
                            rollback_error = exc
                if rollback_error is not None:
                    raise RuntimeError("gradient copy rollback failed") from rollback_error
                raise original

    def reset(self):
        """Discard all accumulated micro-batches without touching live gradients."""
        with self._lock:
            self._averages = tuple(
                np.zeros(shape, dtype=np.float64) for shape in self._shapes
            )
            self._total_weight = 0.0
            self._accumulation_count = 0

    def state_dict(self):
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "total_weight": self._total_weight,
                "accumulation_count": self._accumulation_count,
                "averages": [np.array(value, copy=True) for value in self._averages],
            }

    def load_state_dict(self, state):
        """Transactionally restore accumulator state without touching live gradients."""
        if not isinstance(state, Mapping):
            raise TypeError("gradient accumulator state must be a mapping")

        version = state.get("version")
        if isinstance(version, (bool, np.bool_)) or version != _STATE_VERSION:
            raise ValueError("unsupported gradient accumulator state version")
        state_type = state.get("type")
        if not isinstance(state_type, str) or state_type != _STATE_TYPE:
            raise ValueError("gradient accumulator state type mismatch")

        total_weight = _nonnegative_real(state.get("total_weight"), "total_weight")
        count = _nonnegative_int(state.get("accumulation_count"), "accumulation_count")
        if (count == 0) != (total_weight == 0.0):
            raise ValueError("gradient accumulator count/weight state is inconsistent")

        averages = state.get("averages")
        if not isinstance(averages, (list, tuple)):
            raise TypeError("gradient accumulator averages must be a list or tuple")
        if len(averages) != len(self._parameters):
            raise ValueError("gradient accumulator average count mismatch")

        converted = []
        for index, (value, shape) in enumerate(zip(averages, self._shapes)):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"gradient accumulator average {index} must be a NumPy array")
            if value.shape != shape:
                raise ValueError(
                    f"gradient accumulator average {index} shape mismatch: expected {shape}, got {value.shape}"
                )
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(
                    f"gradient accumulator average {index} must contain floating-point values"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"gradient accumulator average {index} must contain only finite values"
                )
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                snapshot = np.asarray(value, dtype=np.float64)
            if not np.isfinite(snapshot).all():
                raise ValueError(
                    f"gradient accumulator average {index} must fit in float64"
                )
            converted.append(np.array(snapshot, copy=True))

        if count == 0 and any(np.any(value != 0.0) for value in converted):
            raise ValueError("empty gradient accumulator state must contain zero averages")

        with self._lock:
            self._validate_binding()
            self._averages = tuple(converted)
            self._total_weight = total_weight
            self._accumulation_count = count
