"""Overflow-stable factored second-moment optimizer for tiny-transformer.

This module implements an explicit-learning-rate Adafactor variant. Matrix and
higher-rank parameters factor the last two dimensions into row/column second
moments; vectors and scalars use an unfactored second moment. Optimizer state is
stored as a root scale plus normalized moment buffers so finite float64
gradients do not need to be squared in their raw magnitude domain.
"""

from collections.abc import Iterable, Mapping
from numbers import Integral, Real
import sys
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "Adafactor"
_MAX_STEP = sys.maxsize
_GLOBAL_LOCK = threading.RLock()


def _finite_real(name, value, *, positive=False, lower=None, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and normalized < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and normalized >= upper:
        raise ValueError(f"{name} must be less than {upper}")
    return normalized


def _bool_flag(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _nonnegative_int(name, value, *, maximum=_MAX_STEP):
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
            raise ValueError("optimizer parameters must not contain duplicate Tensor identities")
        seen.add(marker)
    return materialized


def _is_factored_shape(shape):
    return len(shape) >= 2 and shape[-2] > 0 and shape[-1] > 0


def _empty_state(shape):
    if _is_factored_shape(shape):
        return {
            "kind": "factored",
            "step": 0,
            "scale": 0.0,
            "row": np.zeros(shape[:-1], dtype=np.float64),
            "col": np.zeros(shape[:-2] + shape[-1:], dtype=np.float64),
        }
    return {
        "kind": "full",
        "step": 0,
        "scale": 0.0,
        "v": np.zeros(shape, dtype=np.float64),
    }


def _copy_state(state):
    if state["kind"] == "factored":
        return {
            "kind": "factored",
            "step": state["step"],
            "scale": state["scale"],
            "row": state["row"].copy(),
            "col": state["col"].copy(),
        }
    return {
        "kind": "full",
        "step": state["step"],
        "scale": state["scale"],
        "v": state["v"].copy(),
    }


def _float64_array_copy(array, *, name, shape=None, nonnegative=False):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if shape is not None and array.shape != shape:
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
            result = np.asarray(array, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must fit float64")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


def _max_abs(array):
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _canonicalize_buffers(scale, buffers):
    peak = 0.0
    for buffer in buffers:
        if buffer.size:
            peak = max(peak, float(np.max(buffer)))
    if peak == 0.0:
        return 0.0, tuple(np.zeros_like(buffer) for buffer in buffers)

    root = float(np.sqrt(peak))
    candidate_scale = scale * root
    if candidate_scale == 0.0 and scale > 0.0:
        # The physical root moment is smaller than the least subnormal. Keep the
        # old root scale and the tiny normalized buffers rather than erasing it.
        return scale, tuple(buffer.copy() for buffer in buffers)
    if not np.isfinite(candidate_scale):
        raise ValueError("Adafactor second-moment scale is not representable")

    normalized = tuple(buffer / peak for buffer in buffers)
    for buffer in normalized:
        if not np.all(np.isfinite(buffer)):
            raise ValueError("Adafactor normalized second moment is not finite")
    return candidate_scale, normalized


def _advance_full(state, gradient, beta2, sqrt_eps):
    old_scale = state["scale"]
    gradient_scale = _max_abs(gradient)
    common_scale = max(old_scale, gradient_scale, sqrt_eps)

    old_ratio = 0.0 if old_scale == 0.0 else old_scale / common_scale
    eps_ratio = sqrt_eps / common_scale
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        g_scaled = gradient / common_scale
        sample = g_scaled * g_scaled + eps_ratio * eps_ratio
        candidate = (
            beta2 * state["v"] * (old_ratio * old_ratio)
            + (1.0 - beta2) * sample
        )
    scale, (candidate,) = _canonicalize_buffers(common_scale, (candidate,))

    if scale == 0.0:
        direction = np.zeros_like(gradient)
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            g_scaled = gradient / scale
            denominator = np.sqrt(candidate)
            direction = np.divide(
                g_scaled,
                denominator,
                out=np.zeros_like(g_scaled),
                where=denominator > 0.0,
            )

    return {
        "kind": "full",
        "step": state["step"] + 1,
        "scale": scale,
        "v": candidate,
    }, direction


def _advance_factored(state, gradient, beta2, sqrt_eps):
    old_scale = state["scale"]
    gradient_scale = _max_abs(gradient)
    common_scale = max(old_scale, gradient_scale, sqrt_eps)

    old_ratio = 0.0 if old_scale == 0.0 else old_scale / common_scale
    eps_ratio = sqrt_eps / common_scale
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        g_scaled = gradient / common_scale
        squared = g_scaled * g_scaled + eps_ratio * eps_ratio
        sample_row = np.mean(squared, axis=-1)
        sample_col = np.mean(squared, axis=-2)
        old_weight = beta2 * (old_ratio * old_ratio)
        new_weight = 1.0 - beta2
        row = old_weight * state["row"] + new_weight * sample_row
        col = old_weight * state["col"] + new_weight * sample_col

    scale, (row, col) = _canonicalize_buffers(common_scale, (row, col))

    if scale == 0.0:
        direction = np.zeros_like(gradient)
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            g_scaled = gradient / scale
            row_mean = np.mean(row, axis=-1, keepdims=True)
            row_ratio = np.divide(
                row,
                row_mean,
                out=np.ones_like(row),
                where=row_mean > 0.0,
            )
            row_factor = np.divide(
                1.0,
                np.sqrt(row_ratio),
                out=np.zeros_like(row_ratio),
                where=row_ratio > 0.0,
            )
            col_root = np.sqrt(col)
            direction = np.divide(
                g_scaled,
                np.expand_dims(col_root, axis=-2),
                out=np.zeros_like(g_scaled),
                where=np.expand_dims(col_root > 0.0, axis=-2),
            )
            direction = direction * np.expand_dims(row_factor, axis=-1)

    return {
        "kind": "factored",
        "step": state["step"] + 1,
        "scale": scale,
        "row": row,
        "col": col,
    }, direction


def _clip_direction(direction, clip_threshold):
    if direction.size == 0:
        return direction
    scale = _max_abs(direction)
    if not np.isfinite(scale):
        raise ValueError("Adafactor update direction is not finite")
    if scale == 0.0 or scale <= clip_threshold:
        return direction

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = direction / scale
        rms_normalized = float(np.sqrt(np.mean(normalized * normalized)))
    if rms_normalized == 0.0:
        return direction
    multiplier = (clip_threshold / scale) / rms_normalized
    if multiplier >= 1.0:
        return direction
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        clipped = direction * multiplier
    if not np.all(np.isfinite(clipped)):
        raise ValueError("Adafactor clipped update is not finite")
    return clipped


def _validate_live_binding(parameter, expected_shape, index):
    if parameter.shape != expected_shape:
        raise ValueError(
            f"parameter {index} shape changed from {expected_shape} to {parameter.shape}"
        )
    data = parameter.data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"parameter {index} data must be a NumPy array")
    return data


def _snapshot_gradient(parameter, expected_shape, index):
    gradient = parameter.grad
    if gradient is None:
        return None
    if not parameter.requires_grad:
        raise ValueError(f"parameter {index} is frozen but still has a gradient")
    return _float64_array_copy(
        gradient,
        name=f"gradient for parameter {index}",
        shape=expected_shape,
    )


def _validate_write_storage(destinations, candidates, active_indexes):
    for index in active_indexes:
        destination = destinations[index]
        candidate = candidates[index]
        if not np.array_equal(destination, candidate) and not destination.flags.writeable:
            raise ValueError(f"parameter {index} data must be writable")

    for right_position, right_index in enumerate(active_indexes):
        for left_index in active_indexes[:right_position]:
            left = destinations[left_index]
            right = destinations[right_index]
            needs_write = (
                not np.array_equal(left, candidates[left_index])
                or not np.array_equal(right, candidates[right_index])
            )
            if not needs_write:
                continue
            try:
                overlaps = np.shares_memory(left, right)
            except ValueError as exc:
                raise ValueError("Adafactor parameter storage overlap could not be determined") from exc
            if overlaps:
                raise ValueError(
                    "Adafactor parameter data storage must not overlap between "
                    f"parameters {left_index} and {right_index}"
                )


def _restore_parameter(parameter, original):
    live = parameter.data
    if isinstance(live, np.ndarray) and live.shape == original.shape and live.flags.writeable:
        live[...] = original
    else:
        parameter.data = original
    if not np.array_equal(parameter.data, original):
        raise RuntimeError("rollback postcondition failed")


def _normalize_loaded_state(raw, shape, index):
    if not isinstance(raw, Mapping):
        raise TypeError(f"Adafactor state[{index}] must be a mapping")
    expected_kind = "factored" if _is_factored_shape(shape) else "full"
    kind = raw.get("kind")
    if kind != expected_kind:
        raise ValueError(
            f"Adafactor state[{index}] kind must be {expected_kind!r}"
        )
    step = _nonnegative_int(f"Adafactor state[{index}] step", raw.get("step"))
    scale = _finite_real(f"Adafactor state[{index}] scale", raw.get("scale"), lower=0.0)

    if kind == "factored":
        row_shape = shape[:-1]
        col_shape = shape[:-2] + shape[-1:]
        row = _float64_array_copy(
            raw.get("row"),
            name=f"Adafactor state[{index}] row",
            shape=row_shape,
            nonnegative=True,
        )
        col = _float64_array_copy(
            raw.get("col"),
            name=f"Adafactor state[{index}] col",
            shape=col_shape,
            nonnegative=True,
        )
        buffers = (row, col)
    else:
        v = _float64_array_copy(
            raw.get("v"),
            name=f"Adafactor state[{index}] v",
            shape=shape,
            nonnegative=True,
        )
        buffers = (v,)

    has_moment = any(buffer.size and np.any(buffer != 0.0) for buffer in buffers)
    if step == 0:
        if scale != 0.0 or has_moment:
            raise ValueError(f"unused Adafactor state[{index}] must be empty")
    else:
        if scale <= 0.0 or not has_moment:
            raise ValueError(f"active Adafactor state[{index}] must contain a moment")
        scale, buffers = _canonicalize_buffers(scale, buffers)
        if scale <= 0.0:
            raise ValueError(f"active Adafactor state[{index}] must contain a moment")

    if kind == "factored":
        return {
            "kind": kind,
            "step": step,
            "scale": scale,
            "row": buffers[0],
            "col": buffers[1],
        }
    return {
        "kind": kind,
        "step": step,
        "scale": scale,
        "v": buffers[0],
    }


class Adafactor:
    """Explicit-learning-rate Adafactor with overflow-stable moment storage."""

    def __init__(
        self,
        parameters,
        *,
        lr=1e-3,
        beta2=0.999,
        eps=1e-30,
        clip_threshold=1.0,
    ):
        self.parameters = _materialize_parameters(parameters)
        self.lr = _finite_real("lr", lr, positive=True)
        self.beta2 = _finite_real("beta2", beta2, lower=0.0, upper=1.0)
        self.eps = _finite_real("eps", eps, positive=True)
        self.clip_threshold = _finite_real(
            "clip_threshold", clip_threshold, positive=True
        )
        self._shapes = tuple(parameter.shape for parameter in self.parameters)
        self._states = tuple(_empty_state(shape) for shape in self._shapes)

    @property
    def steps(self):
        with _GLOBAL_LOCK:
            return tuple(state["step"] for state in self._states)

    def step(self):
        """Apply one transactional Adafactor update to every active parameter."""
        with _GLOBAL_LOCK:
            sqrt_eps = float(np.sqrt(self.eps))
            destinations = []
            gradients = []
            for index, (parameter, shape) in enumerate(
                zip(self.parameters, self._shapes)
            ):
                data = _validate_live_binding(parameter, shape, index)
                destinations.append(data)
                gradient = _snapshot_gradient(parameter, shape, index)
                if gradient is not None and not np.all(np.isfinite(data)):
                    raise ValueError(
                        f"parameter {index} must contain only finite values before step()"
                    )
                gradients.append(gradient)

            candidate_states = list(self._states)
            candidates = [np.array(data, copy=True, subok=False) for data in destinations]
            active_indexes = []

            for index, (state, gradient, data) in enumerate(
                zip(self._states, gradients, destinations)
            ):
                if gradient is None or gradient.size == 0:
                    continue
                if state["step"] >= _MAX_STEP:
                    raise OverflowError(
                        f"Adafactor parameter {index} reached the supported step maximum"
                    )
                try:
                    if state["kind"] == "factored":
                        new_state, direction = _advance_factored(
                            state, gradient, self.beta2, sqrt_eps
                        )
                    else:
                        new_state, direction = _advance_full(
                            state, gradient, self.beta2, sqrt_eps
                        )
                    direction = _clip_direction(direction, self.clip_threshold)
                    with np.errstate(
                        over="raise", invalid="raise", divide="raise", under="ignore"
                    ):
                        candidate = np.asarray(data, dtype=np.float64) - self.lr * direction
                except FloatingPointError as exc:
                    raise ValueError(
                        f"Adafactor update for parameter {index} is not representable"
                    ) from exc
                if not np.all(np.isfinite(candidate)):
                    raise ValueError(
                        f"Adafactor update for parameter {index} is not representable"
                    )
                candidate_states[index] = new_state
                candidates[index] = np.array(candidate, copy=True, subok=False)
                active_indexes.append(index)

            _validate_write_storage(
                tuple(destinations), tuple(candidates), tuple(active_indexes)
            )
            originals = tuple(
                np.array(data, copy=True, subok=False) for data in destinations
            )
            attempted = []
            try:
                for index in active_indexes:
                    destination = destinations[index]
                    candidate = candidates[index]
                    if np.array_equal(destination, candidate):
                        continue
                    attempted.append(index)
                    destination[...] = candidate
                    if not np.array_equal(self.parameters[index].data, candidate):
                        raise RuntimeError(
                            f"parameter {index} rejected Adafactor update"
                        )
            except Exception:
                rollback_error = None
                for index in reversed(attempted):
                    try:
                        _restore_parameter(self.parameters[index], originals[index])
                    except Exception as exc:  # pragma: no cover - injected failure path
                        if rollback_error is None:
                            rollback_error = exc
                        continue
                if rollback_error is not None:
                    raise RuntimeError("Adafactor parameter rollback failed") from rollback_error
                raise

            self._states = tuple(candidate_states)
        return None

    def zero_grad(self, set_to_none=False):
        """Clear bound gradients using scalar-safe whole-array assignment."""
        set_to_none = _bool_flag("set_to_none", set_to_none)
        with _GLOBAL_LOCK:
            for parameter in self.parameters:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad[...] = 0.0

    def state_dict(self):
        """Return independent checkpoint state."""
        with _GLOBAL_LOCK:
            states = []
            for state in self._states:
                if state["kind"] == "factored":
                    states.append(
                        {
                            "kind": "factored",
                            "step": state["step"],
                            "scale": state["scale"],
                            "row": state["row"].copy(),
                            "col": state["col"].copy(),
                        }
                    )
                else:
                    states.append(
                        {
                            "kind": "full",
                            "step": state["step"],
                            "scale": state["scale"],
                            "v": state["v"].copy(),
                        }
                    )
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "lr": self.lr,
                "beta2": self.beta2,
                "eps": self.eps,
                "clip_threshold": self.clip_threshold,
                "states": states,
            }

    def load_state_dict(self, state):
        """Validate and transactionally replace optimizer state."""
        if not isinstance(state, Mapping):
            raise TypeError("Adafactor state must be a mapping")
        version = _nonnegative_int("Adafactor version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported Adafactor version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"Adafactor type must be {_STATE_TYPE!r}")

        lr = _finite_real("Adafactor lr", state.get("lr"), positive=True)
        beta2 = _finite_real(
            "Adafactor beta2", state.get("beta2"), lower=0.0, upper=1.0
        )
        eps = _finite_real("Adafactor eps", state.get("eps"), positive=True)
        clip_threshold = _finite_real(
            "Adafactor clip_threshold", state.get("clip_threshold"), positive=True
        )

        raw_states = state.get("states")
        if isinstance(raw_states, (str, bytes)) or not isinstance(raw_states, Iterable):
            raise TypeError("Adafactor states must be an iterable of mappings")
        raw_states = tuple(raw_states)
        if len(raw_states) != len(self.parameters):
            raise ValueError(
                "Adafactor state count mismatch: "
                f"expected {len(self.parameters)}, got {len(raw_states)}"
            )
        normalized_states = tuple(
            _normalize_loaded_state(raw, shape, index)
            for index, (raw, shape) in enumerate(zip(raw_states, self._shapes))
        )

        with _GLOBAL_LOCK:
            for index, (parameter, shape) in enumerate(
                zip(self.parameters, self._shapes)
            ):
                _validate_live_binding(parameter, shape, index)
            self.lr = lr
            self.beta2 = beta2
            self.eps = eps
            self.clip_threshold = clip_threshold
            self._states = normalized_states
        return self
