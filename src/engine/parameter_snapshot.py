"""Transactional parameter snapshots and temporary model-value installation."""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from numbers import Integral
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "ParameterSnapshot"
_LOCK = threading.RLock()


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
    return materialized


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _float64_array_copy(array, *, name, expected_shape=None):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")

    if array.dtype.itemsize > np.dtype(np.float64).itemsize:
        limit = np.array(np.finfo(np.float64).max, dtype=array.dtype)
        with np.errstate(over="ignore", invalid="raise", under="ignore"):
            if np.any(np.abs(array) > limit):
                raise ValueError(f"{name} must fit float64")

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.asarray(array, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must fit float64")
    return result


def _arrays_equal(left, right):
    return left.shape == right.shape and np.array_equal(left, right)


def _materialize_values(values, shapes, *, prefix):
    if len(shapes) == 1 and isinstance(values, np.ndarray):
        materialized = (values,)
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{prefix} must be an iterable of NumPy arrays")
        try:
            materialized = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{prefix} must be an iterable of NumPy arrays") from exc

    if len(materialized) != len(shapes):
        raise ValueError(f"{prefix} count must match bound parameters")

    return tuple(
        _float64_array_copy(
            value,
            name=f"{prefix} {index}",
            expected_shape=shape,
        )
        for index, (value, shape) in enumerate(zip(materialized, shapes))
    )


def _snapshot_live(parameters, shapes):
    snapshots = []
    for index, (parameter, shape) in enumerate(zip(parameters, shapes)):
        if parameter.shape != shape:
            raise ValueError(
                f"parameter {index} shape changed from {shape} to {parameter.shape}"
            )
        snapshots.append(
            _float64_array_copy(
                parameter.data,
                name=f"parameter {index} data",
                expected_shape=shape,
            )
        )
    return tuple(snapshots)


def _validate_independent_storage(destinations):
    for right_index, right in enumerate(destinations):
        for left_index in range(right_index):
            try:
                overlaps = np.shares_memory(destinations[left_index], right)
            except ValueError as exc:
                raise ValueError(
                    "parameter storage overlap could not be determined"
                ) from exc
            if overlaps:
                raise ValueError(
                    "parameter data storage must not overlap between "
                    f"parameters {left_index} and {right_index}"
                )


def _preflight_destinations(parameters, shapes, targets):
    destinations = []
    for index, (parameter, shape, target) in enumerate(
        zip(parameters, shapes, targets)
    ):
        if parameter.shape != shape:
            raise ValueError(
                f"parameter {index} shape changed from {shape} to {parameter.shape}"
            )
        data = parameter.data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"parameter {index} data must be a NumPy array")
        if data.shape != shape:
            raise ValueError(
                f"parameter {index} data shape changed from {shape} to {data.shape}"
            )
        if not data.flags.writeable and not _arrays_equal(data, target):
            raise ValueError(f"parameter {index} data must be writable")
        destinations.append(data)
    destinations = tuple(destinations)
    _validate_independent_storage(destinations)
    return destinations


def _restore_one(parameter, entry):
    live = parameter.data
    if isinstance(live, np.ndarray) and live.shape == entry.shape and live.flags.writeable:
        if not _arrays_equal(live, entry):
            live[...] = entry
    else:
        parameter.data = entry
    if not _arrays_equal(np.asarray(parameter.data), entry):
        raise RuntimeError("parameter restoration postcondition failed")


def _install_locked(parameters, shapes, targets):
    destinations = _preflight_destinations(parameters, shapes, targets)
    originals = tuple(np.array(data, dtype=np.float64, copy=True, subok=False) for data in destinations)
    attempted = []
    try:
        for index, (parameter, data, target) in enumerate(
            zip(parameters, destinations, targets)
        ):
            if _arrays_equal(data, target):
                continue
            attempted.append(index)
            data[...] = target
            if not _arrays_equal(np.asarray(parameter.data), target):
                raise RuntimeError(f"parameter {index} rejected snapshot values")
    except Exception:
        rollback_error = None
        for index in reversed(attempted):
            try:
                _restore_one(parameters[index], originals[index])
            except Exception as exc:  # pragma: no cover - injected failure path
                if rollback_error is None:
                    rollback_error = exc
                continue
        if rollback_error is not None:
            raise RuntimeError("parameter snapshot rollback failed") from rollback_error
        raise
    return len(attempted)


class ParameterSnapshot:
    """Store an ordered parameter-value snapshot and install it transactionally."""

    def __init__(self, parameters, *, values=None):
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        with _LOCK:
            if values is None:
                self._values = _snapshot_live(self._parameters, self._shapes)
            else:
                self._values = _materialize_values(
                    values, self._shapes, prefix="snapshot value"
                )

    @property
    def parameters(self):
        return self._parameters

    @property
    def parameter_count(self):
        return len(self._parameters)

    def values(self):
        """Return deep independent copies of the stored parameter values."""
        with _LOCK:
            return tuple(value.copy() for value in self._values)

    def capture(self):
        """Replace the stored snapshot with the current live parameter values."""
        with _LOCK:
            candidates = _snapshot_live(self._parameters, self._shapes)
            self._values = candidates
            return self

    def restore(self):
        """Transactionally install the stored snapshot into the bound parameters."""
        with _LOCK:
            return _install_locked(self._parameters, self._shapes, self._values)

    @contextmanager
    def installed(self):
        """Temporarily install the snapshot and restore entry values on exit."""
        with _LOCK:
            entry_values = _snapshot_live(self._parameters, self._shapes)
            _install_locked(self._parameters, self._shapes, self._values)
            try:
                yield self
            finally:
                restoration_error = None
                for parameter, entry in zip(self._parameters, entry_values):
                    try:
                        _restore_one(parameter, entry)
                    except Exception as exc:  # pragma: no cover - injected failure path
                        if restoration_error is None:
                            restoration_error = exc
                        continue
                if restoration_error is not None:
                    raise RuntimeError(
                        "parameter snapshot restoration failed"
                    ) from restoration_error

    def state_dict(self):
        """Return independent checkpoint state for the stored snapshot."""
        with _LOCK:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "values": [value.copy() for value in self._values],
            }

    def load_state_dict(self, state):
        """Validate and replace stored snapshot state without touching the model."""
        if not isinstance(state, Mapping):
            raise TypeError("parameter snapshot state must be a mapping")
        version = _nonnegative_int(
            "parameter snapshot version", state.get("version")
        )
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported parameter snapshot version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"parameter snapshot type must be {_STATE_TYPE!r}")
        raw_values = state.get("values")
        candidates = _materialize_values(
            raw_values,
            self._shapes,
            prefix="parameter snapshot state value",
        )

        with _LOCK:
            for index, (parameter, shape) in enumerate(
                zip(self._parameters, self._shapes)
            ):
                if parameter.shape != shape:
                    raise ValueError(
                        f"parameter {index} shape changed from {shape} to {parameter.shape}"
                    )
            self._values = candidates
        return self
