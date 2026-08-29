"""Overflow-stable LAMB optimizer for tiny-transformer.

The optimizer stores bias-corrected first moments directly and stores every
bias-corrected second moment as a root scale plus a normalized non-negative
buffer. This avoids squaring finite float64 gradients in their original
magnitude domain.

LAMB's trust-ratio step is evaluated without materializing ``||p|| / ||u||``.
When both parameter and update norms are non-zero, the exact rule
``lr * (||p|| / ||u||) * u`` is rearranged as
``lr * ||p|| * (u / ||u||)``. The parameter norm stays in a binary-exponent
scaled descriptor until the final per-component step is reconstructed.
"""

from collections.abc import Iterable, Mapping
from numbers import Integral, Real
import math
import sys
import threading

import numpy as np

from .tensor import Tensor


_STATE_VERSION = 1
_STATE_TYPE = "LAMB"
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


def _validate_betas(betas, name):
    try:
        values = tuple(betas)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real numbers") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    return (
        _finite_real(f"{name}[0]", values[0], lower=0.0, upper=1.0),
        _finite_real(f"{name}[1]", values[1], lower=0.0, upper=1.0),
    )


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


def _convex_update(old, new, new_weight):
    if new_weight == 0.0:
        return old.copy()
    if new_weight == 1.0:
        return new.copy()
    if old.size == 0:
        return old.copy()

    old_flat = old.reshape(-1)
    new_flat = new.reshape(-1)
    result = np.empty_like(old_flat, dtype=np.float64)
    same_sign = np.signbit(old_flat) == np.signbit(new_flat)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        if np.any(same_sign):
            left = old_flat[same_sign]
            right = new_flat[same_sign]
            result[same_sign] = left + new_weight * (right - left)
        if np.any(~same_sign):
            left = old_flat[~same_sign]
            right = new_flat[~same_sign]
            result[~same_sign] = (1.0 - new_weight) * left + new_weight * right
    if not np.all(np.isfinite(result)):
        raise ValueError("LAMB first moment is not representable")
    return result.reshape(old.shape)


def _corrected_ema_new_weight(beta, step):
    if step <= 0:
        raise ValueError("LAMB corrected EMA step must be positive")
    if beta == 0.0 or step == 1:
        return 1.0
    bias_correction = 1.0 - beta ** step
    if bias_correction <= 0.0:
        raise ValueError("LAMB bias correction is not representable")
    new_weight = (1.0 - beta) / bias_correction
    if not (0.0 < new_weight <= 1.0):
        raise ValueError("LAMB corrected EMA weight is invalid")
    return new_weight


def _canonicalize_v(scale, buffer):
    peak = float(np.max(buffer)) if buffer.size else 0.0
    if peak == 0.0:
        return 0.0, np.zeros_like(buffer)
    if scale <= 0.0:
        raise ValueError("LAMB active second moment requires a positive scale")

    root = float(np.sqrt(peak))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        candidate_scale = scale * root
    if candidate_scale == 0.0:
        return scale, buffer.copy()
    if not np.isfinite(candidate_scale):
        raise ValueError("LAMB second-moment scale is not representable")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = buffer / peak
    if not np.all(np.isfinite(normalized)):
        raise ValueError("LAMB normalized second moment is not finite")
    return candidate_scale, normalized


def _advance_second_moment(scale, buffer, gradient, new_weight):
    gradient_scale = _max_abs(gradient)
    common_scale = max(scale, gradient_scale)
    if common_scale == 0.0:
        return 0.0, np.zeros_like(buffer)

    old_ratio = 0.0 if scale == 0.0 else scale / common_scale
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        g_scaled = gradient / common_scale
        candidate = (
            (1.0 - new_weight) * buffer * (old_ratio * old_ratio)
            + new_weight * (g_scaled * g_scaled)
        )
    if not np.all(np.isfinite(candidate)) or np.any(candidate < 0.0):
        raise ValueError("LAMB second moment is not representable")
    return _canonicalize_v(common_scale, candidate)


def _adam_direction(first_moment, second_scale, second_buffer, eps):
    if first_moment.size == 0:
        return first_moment.copy()
    common_scale = max(second_scale, eps)
    second_ratio = 0.0 if second_scale == 0.0 else second_scale / common_scale
    eps_ratio = eps / common_scale
    try:
        with np.errstate(
            over="raise", invalid="raise", divide="raise", under="ignore"
        ):
            root = np.sqrt(second_buffer)
            denominator = second_ratio * root + eps_ratio
            numerator = first_moment / common_scale
            direction = numerator / denominator
    except FloatingPointError as exc:
        raise ValueError("LAMB Adam direction is not representable") from exc
    if not np.all(np.isfinite(direction)):
        raise ValueError("LAMB Adam direction is not representable")
    return direction


def _descriptor_from_float(value):
    value = float(value)
    if value < 0.0 or not np.isfinite(value):
        raise ValueError("scaled magnitude must be finite and non-negative")
    if value == 0.0:
        return (0.0, 0)
    return math.frexp(value)


def _descriptor_multiply(left, right):
    if left[0] == 0.0 or right[0] == 0.0:
        return (0.0, 0)
    mantissa, adjust = math.frexp(left[0] * right[0])
    return (mantissa, left[1] + right[1] + adjust)


def _descriptor_multiply_float(descriptor, value):
    return _descriptor_multiply(descriptor, _descriptor_from_float(value))


def _descriptor_compare(left, right):
    if left[0] == 0.0:
        return 0 if right[0] == 0.0 else -1
    if right[0] == 0.0:
        return 1
    if left[1] != right[1]:
        return -1 if left[1] < right[1] else 1
    if left[0] == right[0]:
        return 0
    return -1 if left[0] < right[0] else 1


def _descriptor_ratio(smaller, larger):
    if smaller[0] == 0.0:
        return 0.0
    if _descriptor_compare(smaller, larger) > 0:
        raise ValueError("scaled ratio requires ordered magnitudes")
    if smaller == larger:
        return 1.0
    ratio = math.ldexp(
        smaller[0] / larger[0], smaller[1] - larger[1]
    )
    return min(1.0, max(0.0, ratio))


def _norm_descriptor(array):
    scale = _max_abs(array)
    if scale == 0.0:
        return (0.0, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = array / scale
        root = float(np.sqrt(np.sum(normalized * normalized)))
    return _descriptor_multiply(
        _descriptor_from_float(scale), _descriptor_from_float(root)
    )


def _unit_direction(array):
    scale = _max_abs(array)
    if scale == 0.0:
        return None
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = array / scale
        root = float(np.sqrt(np.sum(normalized * normalized)))
        direction = normalized / root
    if root == 0.0 or not np.all(np.isfinite(direction)):
        raise ValueError("LAMB update direction is not representable")
    return direction


def _scale_array_by_descriptor(array, descriptor):
    if descriptor[0] == 0.0 or array.size == 0:
        return np.zeros_like(array, dtype=np.float64)
    mantissas, exponents = np.frexp(array)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        scaled_mantissas = mantissas * descriptor[0]
        combined_exponents = exponents.astype(np.int64) + descriptor[1]
        result = np.ldexp(scaled_mantissas, combined_exponents)
    if not np.all(np.isfinite(result)):
        raise ValueError("LAMB trusted step is not representable")
    return np.asarray(result, dtype=np.float64)


def _normalized_lamb_update(adam_direction, data, weight_decay):
    if weight_decay == 0.0:
        return adam_direction.copy()

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            direct = adam_direction + weight_decay * data
        if np.all(np.isfinite(direct)):
            return np.asarray(direct, dtype=np.float64)
    except FloatingPointError:
        pass

    adam_scale = _max_abs(adam_direction)
    parameter_scale = _max_abs(data)
    adam_descriptor = _descriptor_from_float(adam_scale)
    decay_descriptor = _descriptor_multiply_float(
        _descriptor_from_float(parameter_scale), weight_decay
    )
    common = (
        adam_descriptor
        if _descriptor_compare(adam_descriptor, decay_descriptor) >= 0
        else decay_descriptor
    )
    if common[0] == 0.0:
        return np.zeros_like(adam_direction)

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized_adam = (
            np.zeros_like(adam_direction)
            if adam_scale == 0.0
            else (adam_direction / adam_scale)
            * _descriptor_ratio(adam_descriptor, common)
        )
        normalized_decay = (
            np.zeros_like(data, dtype=np.float64)
            if parameter_scale == 0.0
            else (np.asarray(data, dtype=np.float64) / parameter_scale)
            * _descriptor_ratio(decay_descriptor, common)
        )
        combined = normalized_adam + normalized_decay
    if not np.all(np.isfinite(combined)):
        raise ValueError("LAMB update is not representable")
    return combined


def _candidate_parameter(data, update_representation, lr):
    data64 = np.asarray(data, dtype=np.float64)
    parameter_norm = _norm_descriptor(data64)
    direction = _unit_direction(update_representation)
    if direction is None:
        return np.array(data64, copy=True)

    try:
        if parameter_norm[0] == 0.0:
            step = _scale_array_by_descriptor(
                update_representation, _descriptor_from_float(lr)
            )
        else:
            step_norm = _descriptor_multiply_float(parameter_norm, lr)
            step = _scale_array_by_descriptor(direction, step_norm)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            candidate = data64 - step
    except FloatingPointError as exc:
        raise ValueError("LAMB parameter update is not representable") from exc
    if not np.all(np.isfinite(candidate)):
        raise ValueError("LAMB parameter update is not representable")
    return np.array(candidate, dtype=np.float64, copy=True)


def _validate_live_binding(parameter, expected_shape, index):
    if parameter.shape != expected_shape:
        raise ValueError(
            f"parameter {index} shape changed from {expected_shape} to {parameter.shape}"
        )
    if not isinstance(parameter.requires_grad, (bool, np.bool_)):
        raise TypeError(f"parameter {index} requires_grad must be boolean")
    version = getattr(parameter, "_version", None)
    if type(version) is not int:
        raise TypeError(f"parameter {index} version must be a non-negative integer")
    if version < 0:
        raise ValueError(f"parameter {index} version must be a non-negative integer")
    data = parameter.data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"parameter {index} data must be a NumPy array")
    return data


def _snapshot_gradient(parameter, expected_shape, index):
    gradient = parameter.grad
    if gradient is None:
        return None
    if not bool(parameter.requires_grad):
        raise ValueError(f"parameter {index} is frozen but still has a gradient")
    return _float64_array_copy(
        gradient, name=f"gradient for parameter {index}", shape=expected_shape
    )


def _validate_write_storage(destinations, candidates, active_indexes):
    write_indexes = [
        index
        for index in active_indexes
        if not np.array_equal(destinations[index], candidates[index])
    ]
    for index in write_indexes:
        if not destinations[index].flags.writeable:
            raise ValueError(f"parameter {index} data must be writable")

    checked = set()
    for write_index in write_indexes:
        for other_index, other in enumerate(destinations):
            if other_index == write_index:
                continue
            pair = tuple(sorted((write_index, other_index)))
            if pair in checked:
                continue
            checked.add(pair)
            try:
                overlaps = np.shares_memory(destinations[write_index], other)
            except ValueError as exc:
                raise ValueError(
                    "LAMB parameter storage overlap could not be determined"
                ) from exc
            if overlaps:
                raise ValueError(
                    "LAMB parameter data storage must not overlap between "
                    f"parameters {pair[0]} and {pair[1]}"
                )


def _restore_parameter(parameter, original):
    live = parameter.data
    if (
        isinstance(live, np.ndarray)
        and live.shape == original.shape
        and live.flags.writeable
    ):
        live[...] = original
    else:
        parameter.data = original
    if not np.array_equal(parameter.data, original):
        raise RuntimeError("rollback postcondition failed")


def _empty_state(shape):
    return {
        "step": 0,
        "m": np.zeros(shape, dtype=np.float64),
        "v_scale": 0.0,
        "v": np.zeros(shape, dtype=np.float64),
    }


def _copy_state(state):
    return {
        "step": state["step"],
        "m": state["m"].copy(),
        "v_scale": state["v_scale"],
        "v": state["v"].copy(),
    }


def _normalize_loaded_state(raw, shape, index):
    if not isinstance(raw, Mapping):
        raise TypeError(f"LAMB state[{index}] must be a mapping")
    step = _nonnegative_int(f"LAMB state[{index}] step", raw.get("step"))
    first = _float64_array_copy(
        raw.get("m"), name=f"LAMB state[{index}] m", shape=shape
    )
    scale = _finite_real(
        f"LAMB state[{index}] v_scale", raw.get("v_scale"), lower=0.0
    )
    second = _float64_array_copy(
        raw.get("v"),
        name=f"LAMB state[{index}] v",
        shape=shape,
        nonnegative=True,
    )

    if step == 0:
        if np.any(first != 0.0) or scale != 0.0 or np.any(second != 0.0):
            raise ValueError(f"unused LAMB state[{index}] must be empty")
        return _empty_state(shape)

    if scale == 0.0:
        if np.any(second != 0.0):
            raise ValueError(
                f"LAMB state[{index}] zero second-moment scale requires zero buffer"
            )
    else:
        scale, second = _canonicalize_v(scale, second)
    return {"step": step, "m": first, "v_scale": scale, "v": second}


class LAMB:
    """Layer-wise Adaptive Moments optimizer with stable trust-ratio math."""

    def __init__(
        self,
        parameters,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
    ):
        self.parameters = _materialize_parameters(parameters)
        self.lr = _finite_real("lr", lr, positive=True)
        self.beta1, self.beta2 = _validate_betas(betas, "betas")
        self.eps = _finite_real("eps", eps, positive=True)
        self.weight_decay = _finite_real("weight_decay", weight_decay, lower=0.0)
        self._shapes = tuple(parameter.shape for parameter in self.parameters)
        self._states = tuple(_empty_state(shape) for shape in self._shapes)

    @property
    def steps(self):
        with _GLOBAL_LOCK:
            return tuple(state["step"] for state in self._states)

    def step(self):
        """Apply one transactional LAMB update to every active parameter."""
        with _GLOBAL_LOCK:
            destinations = []
            gradients = []
            for index, (parameter, shape) in enumerate(
                zip(self.parameters, self._shapes)
            ):
                data = _validate_live_binding(parameter, shape, index)
                destinations.append(data)
                gradient = _snapshot_gradient(parameter, shape, index)
                if gradient is not None:
                    _float64_array_copy(
                        data, name=f"parameter {index} data", shape=shape
                    )
                gradients.append(gradient)

            candidate_states = list(self._states)
            candidates = [
                np.array(data, dtype=np.float64, copy=True, subok=False)
                for data in destinations
            ]
            active_indexes = []

            for index, (state, gradient, data) in enumerate(
                zip(self._states, gradients, destinations)
            ):
                if gradient is None or gradient.size == 0:
                    continue
                if state["step"] >= _MAX_STEP:
                    raise OverflowError(
                        f"LAMB parameter {index} reached the supported step maximum"
                    )
                next_step = state["step"] + 1
                first_weight = _corrected_ema_new_weight(self.beta1, next_step)
                second_weight = _corrected_ema_new_weight(self.beta2, next_step)
                try:
                    first = _convex_update(state["m"], gradient, first_weight)
                    second_scale, second = _advance_second_moment(
                        state["v_scale"], state["v"], gradient, second_weight
                    )
                    adam_direction = _adam_direction(
                        first, second_scale, second, self.eps
                    )
                    update = _normalized_lamb_update(
                        adam_direction,
                        np.asarray(data, dtype=np.float64),
                        self.weight_decay,
                    )
                    candidate = _candidate_parameter(data, update, self.lr)
                except FloatingPointError as exc:
                    raise ValueError(
                        f"LAMB update for parameter {index} is not representable"
                    ) from exc

                candidate_states[index] = {
                    "step": next_step,
                    "m": first,
                    "v_scale": second_scale,
                    "v": second,
                }
                candidates[index] = candidate
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
                        raise RuntimeError(f"parameter {index} rejected LAMB update")
            except BaseException:
                rollback_error = None
                for index in reversed(attempted):
                    try:
                        _restore_parameter(self.parameters[index], originals[index])
                    except BaseException as exc:  # pragma: no cover - injected failure
                        if rollback_error is None:
                            rollback_error = exc
                        continue
                if rollback_error is not None:
                    raise RuntimeError("LAMB parameter rollback failed") from rollback_error
                raise

            self._states = tuple(candidate_states)
        return None

    def zero_grad(self, set_to_none=False):
        """Clear gradients using scalar-safe whole-array assignment."""
        set_to_none = _bool_flag("set_to_none", set_to_none)
        with _GLOBAL_LOCK:
            for parameter in self.parameters:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad[...] = 0.0

    def state_dict(self):
        """Return an independent optimizer checkpoint state."""
        with _GLOBAL_LOCK:
            return {
                "version": _STATE_VERSION,
                "type": _STATE_TYPE,
                "lr": self.lr,
                "betas": (self.beta1, self.beta2),
                "eps": self.eps,
                "weight_decay": self.weight_decay,
                "states": [_copy_state(state) for state in self._states],
            }

    def load_state_dict(self, state):
        """Validate and transactionally replace optimizer state."""
        if not isinstance(state, Mapping):
            raise TypeError("LAMB state must be a mapping")
        version = _nonnegative_int("LAMB version", state.get("version"))
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported LAMB version: {version}")
        if state.get("type") != _STATE_TYPE:
            raise ValueError(f"LAMB type must be {_STATE_TYPE!r}")

        lr = _finite_real("LAMB lr", state.get("lr"), positive=True)
        beta1, beta2 = _validate_betas(state.get("betas"), "LAMB betas")
        eps = _finite_real("LAMB eps", state.get("eps"), positive=True)
        weight_decay = _finite_real(
            "LAMB weight_decay", state.get("weight_decay"), lower=0.0
        )

        raw_states = state.get("states")
        if isinstance(raw_states, (str, bytes)) or not isinstance(
            raw_states, Iterable
        ):
            raise TypeError("LAMB states must be an iterable of mappings")
        raw_states = tuple(raw_states)
        if len(raw_states) != len(self.parameters):
            raise ValueError(
                "LAMB state count mismatch: "
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
            self.beta1, self.beta2 = beta1, beta2
            self.eps = eps
            self.weight_decay = weight_decay
            self._states = normalized_states
        return self
