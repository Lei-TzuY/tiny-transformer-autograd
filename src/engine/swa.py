"""Checkpointable stochastic weight averaging for ordered Tensor collections."""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from numbers import Integral
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "StochasticWeightAverage"


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
        identity = id(parameter)
        if identity in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(identity)
    return materialized


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _snapshot_parameter(parameter, expected_shape, index):
    if parameter.shape != expected_shape:
        raise ValueError(
            f"parameter {index} shape changed from {expected_shape} to {parameter.shape}"
        )
    data = parameter.data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"parameter {index} data must be a NumPy array")
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError(f"parameter {index} data must have floating dtype")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"parameter {index} data must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            snapshot = np.asarray(data, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"parameter {index} data must fit float64") from exc
    if not np.all(np.isfinite(snapshot)):
        raise ValueError(f"parameter {index} data must fit float64")
    return snapshot


def _stable_equal_weight_average(previous, current, previous_count):
    """Return the mean of ``previous_count`` samples plus ``current`` safely."""
    total_count = previous_count + 1
    old_fraction = previous_count / total_count
    new_fraction = 1.0 / total_count

    left = np.asarray(previous, dtype=np.float64).reshape(-1)
    right = np.asarray(current, dtype=np.float64).reshape(-1)
    result = np.empty_like(left)

    same_sign = ((left >= 0.0) & (right >= 0.0)) | (
        (left <= 0.0) & (right <= 0.0)
    )
    opposite_sign = ~same_sign

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if np.any(same_sign):
                result[same_sign] = left[same_sign] + new_fraction * (
                    right[same_sign] - left[same_sign]
                )
            if np.any(opposite_sign):
                result[opposite_sign] = (
                    left[opposite_sign] * old_fraction
                    + right[opposite_sign] * new_fraction
                )
    except FloatingPointError as exc:
        raise ValueError("SWA average is not representable as float64") from exc

    if not np.all(np.isfinite(result)):
        raise ValueError("SWA average is not representable as float64")
    return result.reshape(previous.shape)


class StochasticWeightAverage:
    """Maintain the equal-weight mean of explicitly captured model checkpoints."""

    def __init__(self, parameters):
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        self._averages = None
        self._num_averaged = 0
        self._lock = threading.RLock()

    @property
    def parameters(self):
        return self._parameters

    @property
    def num_averaged(self):
        with self._lock:
            return self._num_averaged

    def update(self):
        """Capture the current parameter values as one equally weighted checkpoint."""
        with self._lock:
            snapshots = tuple(
                _snapshot_parameter(parameter, shape, index)
                for index, (parameter, shape) in enumerate(
                    zip(self._parameters, self._shapes)
                )
            )

            if self._num_averaged == 0:
                candidates = tuple(snapshot.copy() for snapshot in snapshots)
            else:
                candidates = tuple(
                    _stable_equal_weight_average(average, snapshot, self._num_averaged)
                    for average, snapshot in zip(self._averages, snapshots)
                )

            self._averages = candidates
            self._num_averaged += 1
            return self._num_averaged

    def averages(self):
        """Return independent copies of the current averaged parameters."""
        with self._lock:
            if self._num_averaged == 0:
                raise RuntimeError("SWA has no averaged checkpoints")
            return tuple(average.copy() for average in self._averages)

    def _preflight_copy_locked(self):
        if self._num_averaged == 0:
            raise RuntimeError("SWA has no averaged checkpoints")
        destinations = []
        for index, (parameter, shape, average) in enumerate(
            zip(self._parameters, self._shapes, self._averages)
        ):
            if parameter.shape != shape:
                raise ValueError(
                    f"parameter {index} shape changed from {shape} to {parameter.shape}"
                )
            data = parameter.data
            if not isinstance(data, np.ndarray):
                raise TypeError(f"parameter {index} data must be a NumPy array")
            if not data.flags.writeable and not np.array_equal(data, average):
                raise ValueError(f"parameter {index} data must be writable")
            destinations.append(data)
        return tuple(destinations)

    def _copy_to_parameters_locked(self):
        destinations = self._preflight_copy_locked()
        originals = tuple(np.array(data, copy=True, subok=False) for data in destinations)
        attempted = []
        try:
            for index, (parameter, data, average) in enumerate(
                zip(self._parameters, destinations, self._averages)
            ):
                if np.array_equal(data, average):
                    continue
                attempted.append(index)
                data[...] = average
                if not np.array_equal(parameter.data, average):
                    raise RuntimeError(f"parameter {index} rejected SWA values")
        except Exception:
            rollback_error = None
            for index in reversed(attempted):
                try:
                    live = self._parameters[index].data
                    if live.shape != originals[index].shape or not live.flags.writeable:
                        self._parameters[index].data = originals[index]
                    else:
                        live[...] = originals[index]
                    if not np.array_equal(self._parameters[index].data, originals[index]):
                        raise RuntimeError("rollback postcondition failed")
                except Exception as exc:  # pragma: no cover - exercised by injected failures
                    rollback_error = exc
                    break
            if rollback_error is not None:
                raise RuntimeError("SWA parameter rollback failed") from rollback_error
            raise
        return len(attempted)

    def copy_to_parameters(self):
        """Transactionally install the current averages into the bound Tensors."""
        with self._lock:
            return self._copy_to_parameters_locked()

    @contextmanager
    def average_parameters(self):
        """Temporarily install averaged parameters and restore entry values on exit."""
        with self._lock:
            if self._num_averaged == 0:
                raise RuntimeError("SWA has no averaged checkpoints")
            entry_values = tuple(
                np.array(parameter.data, copy=True, subok=False)
                for parameter in self._parameters
            )
            self._copy_to_parameters_locked()
            try:
                yield self
            finally:
                restoration_error = None
                for parameter, entry in zip(self._parameters, entry_values):
                    try:
                        live = parameter.data
                        if live.shape == entry.shape and live.flags.writeable:
                            if not np.array_equal(live, entry):
                                live[...] = entry
                        else:
                            parameter.data = entry
                        if not np.array_equal(parameter.data, entry):
                            raise RuntimeError("restoration postcondition failed")
                    except Exception as exc:  # pragma: no cover - defensive failure path
                        restoration_error = exc
                        break
                if restoration_error is not None:
                    raise RuntimeError("SWA parameter restoration failed") from restoration_error

    def reset(self):
        """Forget all captured checkpoints without touching live parameters."""
        with self._lock:
            self._averages = None
            self._num_averaged = 0
            return self

    def state_dict(self):
        """Return independent checkpoint state for the SWA accumulator."""
        with self._lock:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "num_averaged": self._num_averaged,
                "averages": []
                if self._averages is None
                else [average.copy() for average in self._averages],
            }

    def load_state_dict(self, state):
        """Validate and transactionally restore SWA state without model writes."""
        if not isinstance(state, Mapping):
            raise TypeError("SWA state must be a mapping")

        version = _nonnegative_int("SWA version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported SWA version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"SWA type must be {_STATE_TYPE!r}")
        num_averaged = _nonnegative_int("SWA num_averaged", state.get("num_averaged"))

        raw_averages = state.get("averages")
        if isinstance(raw_averages, (str, bytes)) or not isinstance(raw_averages, Iterable):
            raise TypeError("SWA averages must be an iterable of NumPy arrays")
        raw_averages = tuple(raw_averages)

        if num_averaged == 0:
            if raw_averages:
                raise ValueError("empty SWA state must not contain averages")
            candidates = None
        else:
            if len(raw_averages) != len(self._parameters):
                raise ValueError("SWA average count must match bound parameters")
            normalized = []
            for index, (raw, shape) in enumerate(zip(raw_averages, self._shapes)):
                if not isinstance(raw, np.ndarray):
                    raise TypeError(f"SWA average {index} must be a NumPy array")
                if raw.shape != shape:
                    raise ValueError(
                        f"SWA average {index} shape must be {shape}, got {raw.shape}"
                    )
                if not np.issubdtype(raw.dtype, np.floating):
                    raise TypeError(f"SWA average {index} must have floating dtype")
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"SWA average {index} must be finite")
                try:
                    with np.errstate(over="raise", invalid="raise", under="ignore"):
                        candidate = np.asarray(raw, dtype=np.float64).copy()
                except (FloatingPointError, OverflowError) as exc:
                    raise ValueError(f"SWA average {index} must fit float64") from exc
                if not np.all(np.isfinite(candidate)):
                    raise ValueError(f"SWA average {index} must fit float64")
                normalized.append(candidate)
            candidates = tuple(normalized)

        with self._lock:
            for index, (parameter, shape) in enumerate(zip(self._parameters, self._shapes)):
                if parameter.shape != shape:
                    raise ValueError(
                        f"parameter {index} shape changed from {shape} to {parameter.shape}"
                    )
            self._averages = candidates
            self._num_averaged = num_averaged
        return self
