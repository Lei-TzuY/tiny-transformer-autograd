"""Unitwise adaptive gradient clipping with overflow-stable norm comparisons."""

import math
from numbers import Real
import threading

import numpy as np

from .tensor import Tensor


_AGC_LOCK = threading.RLock()
_ZERO_SCALED = (0.0, 0)


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


def _float64_copy(array, *, name, shape, floating_only):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {array.shape}")
    if floating_only:
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{name} must have a floating dtype")
    elif (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
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
    return result


def _scaled_from_float(value):
    if value == 0.0:
        return _ZERO_SCALED
    mantissa, exponent = math.frexp(value)
    return mantissa, exponent


def _scaled_product(scale, factor):
    if scale == 0.0 or factor == 0.0:
        return _ZERO_SCALED
    left_mantissa, left_exponent = math.frexp(scale)
    right_mantissa, right_exponent = math.frexp(factor)
    mantissa, extra = math.frexp(left_mantissa * right_mantissa)
    return mantissa, left_exponent + right_exponent + extra


def _scaled_multiply(value, factor):
    if value[0] == 0.0 or factor == 0.0:
        return _ZERO_SCALED
    factor_mantissa, factor_exponent = math.frexp(factor)
    mantissa, extra = math.frexp(value[0] * factor_mantissa)
    return mantissa, value[1] + factor_exponent + extra


def _scaled_compare(left, right):
    if left[0] == 0.0:
        return 0 if right[0] == 0.0 else -1
    if right[0] == 0.0:
        return 1
    if left[1] != right[1]:
        return -1 if left[1] < right[1] else 1
    if left[0] == right[0]:
        return 0
    return -1 if left[0] < right[0] else 1


def _scaled_max(left, right):
    return left if _scaled_compare(left, right) >= 0 else right


def _scaled_divide_by_float(value, divisor):
    if value[0] == 0.0:
        return 0.0
    divisor_mantissa, divisor_exponent = math.frexp(divisor)
    mantissa, extra = math.frexp(value[0] / divisor_mantissa)
    exponent = value[1] - divisor_exponent + extra
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ValueError("adaptive clipping candidate magnitude is not representable") from exc


def _unit_rows(array):
    if array.ndim == 0:
        return array.reshape((1, 1))
    if array.ndim == 1:
        return array.reshape((1, array.shape[0]))
    unit_count = math.prod(array.shape[:-1])
    return array.reshape((unit_count, array.shape[-1]))


def _norm_descriptor(vector):
    if vector.size == 0:
        return _ZERO_SCALED, 0.0, 0.0
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return _ZERO_SCALED, 0.0, 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        normalized = vector / scale
        root = float(np.sqrt(np.sum(normalized * normalized)))
    if not np.isfinite(root) or root <= 0.0:
        raise ValueError("adaptive clipping unit norm is not finite")
    return _scaled_product(scale, root), scale, root


def _candidate_gradient(parameter_data, gradient_data, gradient_dtype, clip_factor, eps):
    parameter_rows = _unit_rows(parameter_data)
    gradient_rows = _unit_rows(gradient_data)
    candidate = np.array(gradient_data, dtype=gradient_dtype, copy=True)
    candidate_rows = _unit_rows(candidate)
    eps_norm = _scaled_from_float(eps)
    clipped_units = 0

    for index, (parameter_row, gradient_row) in enumerate(
        zip(parameter_rows, gradient_rows)
    ):
        gradient_norm, gradient_scale, gradient_root = _norm_descriptor(gradient_row)
        if gradient_norm[0] == 0.0:
            continue
        parameter_norm, _, _ = _norm_descriptor(parameter_row)
        target_norm = _scaled_multiply(
            _scaled_max(parameter_norm, eps_norm), clip_factor
        )
        if _scaled_compare(gradient_norm, target_norm) <= 0:
            continue

        candidate_peak = _scaled_divide_by_float(target_norm, gradient_root)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            clipped = (gradient_row / gradient_scale) * candidate_peak
        if not np.all(np.isfinite(clipped)):
            raise ValueError("adaptive clipping candidate must contain only finite values")
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                native = np.asarray(clipped, dtype=gradient_dtype)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("adaptive clipping candidate must fit gradient dtype") from exc
        if not np.array_equal(candidate_rows[index], native):
            candidate_rows[index][...] = native
            clipped_units += 1

    return candidate, clipped_units


def _shares_memory(left, right, label):
    try:
        return bool(np.shares_memory(left, right))
    except ValueError as exc:
        raise ValueError(f"{label} storage overlap could not be determined") from exc


def _preflight(destinations, candidates, parameter_data):
    changed = [
        index
        for index, (destination, candidate) in enumerate(zip(destinations, candidates))
        if destination is not None and not np.array_equal(destination, candidate)
    ]

    for index in changed:
        if not destinations[index].flags.writeable:
            raise ValueError(f"gradient for parameter {index} must be writable")

    active = [index for index, destination in enumerate(destinations) if destination is not None]
    for right_position, right_index in enumerate(active):
        for left_index in active[:right_position]:
            if left_index not in changed and right_index not in changed:
                continue
            if _shares_memory(
                destinations[left_index],
                destinations[right_index],
                "gradient",
            ):
                raise ValueError(
                    "gradient storage must not overlap between parameters "
                    f"{left_index} and {right_index} when clipping writes are required"
                )

    for gradient_index in changed:
        gradient = destinations[gradient_index]
        for parameter_index, data in enumerate(parameter_data):
            if _shares_memory(gradient, data, "gradient/parameter"):
                raise ValueError(
                    "gradient storage for parameter "
                    f"{gradient_index} must not overlap parameter {parameter_index} data"
                )
    return changed


def adaptive_clip_grad_(parameters, clip_factor=0.01, eps=1e-3):
    """Clip live gradients relative to unitwise parameter norms.

    Scalars and vectors are treated as one unit. Tensors with rank >= 2 are
    split into vectors along the final axis, matching Linear output rows and
    Embedding rows in this repository. The function returns the number of units
    whose stored gradient values changed.
    """

    clip_factor = _positive_real("clip_factor", clip_factor)
    eps = _positive_real("eps", eps)

    with _AGC_LOCK:
        parameters = _materialize_parameters(parameters)
        parameter_data = []
        destinations = []
        originals = []
        candidates = []
        total_clipped_units = 0

        for index, parameter in enumerate(parameters):
            data = parameter.data
            if not isinstance(data, np.ndarray):
                raise TypeError(f"parameter {index} data must be a NumPy array")
            parameter_data.append(data)

            gradient = parameter.grad
            if gradient is None:
                destinations.append(None)
                originals.append(None)
                candidates.append(None)
                continue
            if not parameter.requires_grad:
                raise ValueError(f"parameter {index} is frozen but still has a gradient")

            data_snapshot = _float64_copy(
                data,
                name=f"parameter {index} data",
                shape=parameter.shape,
                floating_only=False,
            )
            gradient_snapshot = _float64_copy(
                gradient,
                name=f"gradient for parameter {index}",
                shape=parameter.shape,
                floating_only=True,
            )
            candidate, clipped_units = _candidate_gradient(
                data_snapshot,
                gradient_snapshot,
                gradient.dtype,
                clip_factor,
                eps,
            )
            destinations.append(gradient)
            originals.append(np.array(gradient, copy=True))
            candidates.append(candidate)
            total_clipped_units += clipped_units

        changed = _preflight(destinations, candidates, parameter_data)
        attempted = []
        try:
            for index in changed:
                attempted.append(index)
                destinations[index][...] = candidates[index]
                if not np.array_equal(destinations[index], candidates[index]):
                    raise RuntimeError(
                        f"adaptive gradient clipping write failed for parameter {index}"
                    )
        except Exception:
            rollback_error = None
            for index in reversed(attempted):
                try:
                    destinations[index][...] = originals[index]
                    if not np.array_equal(destinations[index], originals[index]):
                        raise RuntimeError("rollback postcondition failed")
                except Exception as exc:  # best-effort cleanup across every attempt
                    if rollback_error is None:
                        rollback_error = exc
            if rollback_error is not None:
                raise RuntimeError("adaptive gradient clipping rollback failed") from rollback_error
            raise

        return total_clipped_units
