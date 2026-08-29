"""Finite-difference parameter-space directional-curvature diagnostics."""

from collections.abc import Iterable
from fractions import Fraction
from numbers import Real
import inspect
import threading

import numpy as np

from .grad_mode import no_grad
from .tensor import Tensor


_PROBE_LOCK = threading.RLock()


def _validate_step(step):
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real):
        raise TypeError("step must be a positive real number")
    try:
        value = float(step)
    except OverflowError as exc:
        raise ValueError("step must fit float64") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("step must be positive and finite")
    return value


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        values = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("parameters must be a Tensor or iterable of Tensors")
        values = tuple(parameters)
    if not values:
        raise ValueError("parameters must contain at least one Tensor")

    seen = set()
    for index, parameter in enumerate(values):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(marker)
        if parameter._children:
            raise ValueError(f"parameter {index} must be a leaf Tensor")
        if not isinstance(parameter.requires_grad, (bool, np.bool_)):
            raise TypeError(f"parameter {index} requires_grad must be boolean")
        if not isinstance(parameter._version, int) or isinstance(parameter._version, bool):
            raise TypeError(f"parameter {index} mutation version must be an integer")
        if parameter._version < 0:
            raise ValueError(f"parameter {index} mutation version must be non-negative")
        data = parameter.data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"parameter {index} data must be a NumPy array")
        if data.shape != parameter.shape:
            raise ValueError(f"parameter {index} data shape is inconsistent")
        if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
            data.dtype, np.complexfloating
        ):
            raise TypeError(f"parameter {index} data must have a real numeric dtype")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"parameter {index} data must contain only finite values")
    return values


def _float64_array_copy(value, *, name, shape):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if value.dtype.itemsize > np.dtype(np.float64).itemsize:
        limit = np.array(np.finfo(np.float64).max, dtype=value.dtype)
        with np.errstate(over="ignore", invalid="raise", under="ignore"):
            if np.any(np.abs(value) > limit):
                raise ValueError(f"{name} must fit float64")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.asarray(value, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must fit float64")
    return result


def _materialize_direction(direction, parameters):
    if len(parameters) == 1 and isinstance(direction, np.ndarray):
        values = (direction,)
    else:
        if isinstance(direction, np.ndarray):
            raise ValueError(
                "multi-parameter direction must be an iterable of NumPy arrays"
            )
        if not isinstance(direction, Iterable):
            raise TypeError("direction must be a NumPy array or iterable of NumPy arrays")
        values = tuple(direction)
    if len(values) != len(parameters):
        raise ValueError(
            f"direction must contain {len(parameters)} arrays, got {len(values)}"
        )

    normalized = tuple(
        _float64_array_copy(
            value,
            name=f"direction[{index}]",
            shape=parameter.shape,
        )
        for index, (parameter, value) in enumerate(zip(parameters, values))
    )
    if not any(np.any(value != 0.0) for value in normalized):
        raise ValueError("direction must contain at least one nonzero element")
    return normalized


def _snapshot_rng_state():
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _snapshot_entries(parameters):
    entries = []
    for parameter in parameters:
        grad = parameter.grad
        grad_copy = (
            np.array(grad, copy=True, subok=False)
            if isinstance(grad, np.ndarray)
            else None
        )
        entries.append(
            {
                "data_ref": parameter.data,
                "data": np.array(parameter.data, dtype=np.float64, copy=True, subok=False),
                "shape": parameter.shape,
                "requires_grad": bool(parameter.requires_grad),
                "version": parameter._version,
                "grad_ref": grad,
                "grad": grad_copy,
                "grad_shape": getattr(parameter, "_grad_shape", None),
            }
        )
    return tuple(entries)


def _exact_affine_scalar(base, direction, coefficient):
    exact = Fraction.from_float(float(base)) + Fraction.from_float(
        float(coefficient)
    ) * Fraction.from_float(float(direction))
    try:
        value = float(exact)
    except OverflowError as exc:
        raise ValueError("finite-difference perturbation is not representable") from exc
    if not np.isfinite(value):
        raise ValueError("finite-difference perturbation is not representable")
    return value


def _affine_candidate(base, direction, coefficient):
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            candidate = base + coefficient * direction
        if np.all(np.isfinite(candidate)):
            return np.array(candidate, dtype=np.float64, copy=True, subok=False)
    except FloatingPointError:
        pass

    result = np.empty(base.shape, dtype=np.float64)
    base_flat = np.asarray(base, dtype=np.float64).reshape(-1)
    direction_flat = np.asarray(direction, dtype=np.float64).reshape(-1)
    result_flat = result.reshape(-1)
    for index, (base_value, direction_value) in enumerate(
        zip(base_flat, direction_flat)
    ):
        result_flat[index] = _exact_affine_scalar(
            base_value, direction_value, coefficient
        )
    return result


def _candidate_values(entries, directions, step):
    plus = []
    minus = []
    for entry, direction in zip(entries, directions):
        plus.append(_affine_candidate(entry["data"], direction, step))
        minus.append(_affine_candidate(entry["data"], direction, -step))
    if not any(
        not np.array_equal(candidate, entry["data"])
        for candidate, entry in zip(plus, entries)
    ):
        raise ValueError("positive perturbation is too small to change any parameter")
    if not any(
        not np.array_equal(candidate, entry["data"])
        for candidate, entry in zip(minus, entries)
    ):
        raise ValueError("negative perturbation is too small to change any parameter")
    if all(np.array_equal(left, right) for left, right in zip(plus, minus)):
        raise ValueError("positive and negative perturbations round to identical values")
    return tuple(plus), tuple(minus)


def _preflight_install(parameters, targets):
    writable = []
    for index, (parameter, target) in enumerate(zip(parameters, targets)):
        data = parameter.data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"parameter {index} data must remain a NumPy array")
        if data.shape != target.shape:
            raise ValueError(f"parameter {index} shape changed during curvature probe")
        needs_write = not np.array_equal(data, target)
        writable.append(needs_write)
        if needs_write and not data.flags.writeable:
            raise ValueError(f"parameter {index} data is read-only")

    for left_index, needs_write in enumerate(writable):
        if not needs_write:
            continue
        left = parameters[left_index].data
        for right_index, parameter in enumerate(parameters):
            if right_index == left_index:
                continue
            try:
                overlaps = np.shares_memory(left, parameter.data)
            except Exception as exc:
                raise ValueError("unable to determine parameter storage overlap") from exc
            if overlaps:
                raise ValueError(
                    "directional curvature writes require non-overlapping bound "
                    f"parameter storage; indexes {left_index} and {right_index} overlap"
                )
    return tuple(writable)


def _install(parameters, targets):
    writable = _preflight_install(parameters, targets)
    for index, (parameter, target, needs_write) in enumerate(
        zip(parameters, targets, writable)
    ):
        if not needs_write:
            continue
        parameter.data[...] = target
        if not np.array_equal(parameter.data, target):
            raise RuntimeError(
                f"parameter {index} did not retain the requested curvature-probe value"
            )


def _restore_entries(parameters, entries):
    first_error = None

    rebuild = set()
    for left_index in range(len(parameters)):
        for right_index in range(left_index + 1, len(parameters)):
            try:
                if np.shares_memory(
                    parameters[left_index].data, parameters[right_index].data
                ):
                    rebuild.add(left_index)
                    rebuild.add(right_index)
            except Exception:
                rebuild.add(left_index)
                rebuild.add(right_index)

    for index, (parameter, entry) in enumerate(zip(parameters, entries)):
        try:
            current = parameter.data
            original = entry["data_ref"]
            if current is not original and index not in rebuild:
                if (
                    isinstance(original, np.ndarray)
                    and original.shape == entry["shape"]
                    and (
                        np.array_equal(original, entry["data"])
                        or original.flags.writeable
                    )
                ):
                    if not np.array_equal(original, entry["data"]):
                        original[...] = entry["data"]
                    parameter._data = original
                    parameter._version += 1
                    current = original
            if (
                index in rebuild
                or not isinstance(current, np.ndarray)
                or current.shape != entry["shape"]
                or not current.flags.writeable
            ):
                parameter.data = entry["data"].copy()
            elif not np.array_equal(current, entry["data"]):
                current[...] = entry["data"]
            if parameter.shape != entry["shape"] or not np.array_equal(
                parameter.data, entry["data"]
            ):
                raise RuntimeError(f"failed to restore parameter {index} data")
        except BaseException as exc:
            if first_error is None:
                first_error = exc

    for parameter, entry in zip(parameters, entries):
        try:
            if (
                not isinstance(parameter.requires_grad, (bool, np.bool_))
                or bool(parameter.requires_grad) != entry["requires_grad"]
            ):
                parameter.requires_grad = entry["requires_grad"]
                parameter._version += 1
            parameter._grad_shape = entry["grad_shape"]
            original_grad = entry["grad_ref"]
            if original_grad is None:
                parameter.grad = None
            else:
                if entry["grad"] is not None and isinstance(original_grad, np.ndarray):
                    if not np.array_equal(original_grad, entry["grad"]):
                        if not original_grad.flags.writeable:
                            raise RuntimeError("original gradient buffer became read-only")
                        original_grad[...] = entry["grad"]
                parameter.grad = original_grad
        except BaseException as exc:
            if first_error is None:
                first_error = exc

    if first_error is not None:
        raise RuntimeError("directional curvature restoration failed") from first_error


def _verify_callback_state(parameters, entries, expected_values, expected_versions):
    for index, (parameter, entry, expected, expected_version) in enumerate(
        zip(parameters, entries, expected_values, expected_versions)
    ):
        if parameter.data is not entry["data_ref"]:
            raise RuntimeError(f"loss callback replaced parameter {index} storage")
        if parameter.shape != entry["shape"]:
            raise RuntimeError(f"loss callback changed parameter {index} shape")
        if not np.array_equal(parameter.data, expected):
            raise RuntimeError(f"loss callback modified parameter {index} values")
        if parameter._version != expected_version:
            raise RuntimeError(f"loss callback modified parameter {index} version")
        if not isinstance(parameter.requires_grad, (bool, np.bool_)) or bool(
            parameter.requires_grad
        ) != entry["requires_grad"]:
            raise RuntimeError(f"loss callback changed parameter {index} trainability")
        if parameter.grad is not entry["grad_ref"]:
            raise RuntimeError(f"loss callback replaced parameter {index} gradient")
        if entry["grad"] is not None:
            current = parameter.grad
            if not isinstance(current, np.ndarray) or not np.array_equal(
                current, entry["grad"]
            ):
                raise RuntimeError(f"loss callback modified parameter {index} gradient")
        if getattr(parameter, "_grad_shape", None) != entry["grad_shape"]:
            raise RuntimeError(f"loss callback changed parameter {index} gradient metadata")


def _loss_scalar(value):
    if isinstance(value, Tensor):
        if value.shape != ():
            raise ValueError("loss callback must return a scalar Tensor or real scalar")
        value = np.asarray(value.data).item()
    elif isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("loss callback must return a scalar Tensor or real scalar")
        value = value.item()

    if inspect.isgenerator(value) or inspect.iscoroutine(value) or inspect.isasyncgen(value):
        if inspect.isgenerator(value) or inspect.iscoroutine(value):
            value.close()
        raise TypeError("loss callback must return synchronously")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("loss callback must return a real scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("loss callback result must fit float64") from exc
    if not np.isfinite(result):
        raise ValueError("loss callback result must be finite")
    return result


def _evaluate(loss_fn, parameters, entries, expected_values, rng_state):
    np.random.set_state(rng_state)
    versions = tuple(parameter._version for parameter in parameters)
    with no_grad():
        value = loss_fn()
    result = _loss_scalar(value)
    _verify_callback_state(parameters, entries, expected_values, versions)
    return result


def _curvature_value(baseline, plus, minus, step):
    numerator = (
        Fraction.from_float(plus)
        - 2 * Fraction.from_float(baseline)
        + Fraction.from_float(minus)
    )
    denominator = Fraction.from_float(step) ** 2
    exact = numerator / denominator
    sign = 0 if exact == 0 else (1 if exact > 0 else -1)
    try:
        value = float(exact)
    except OverflowError:
        return None, True, False, sign
    if not np.isfinite(value):
        return None, True, False, sign
    underflow = exact != 0 and value == 0.0
    return value, False, underflow, sign


def directional_curvature(loss_fn, parameters, direction, *, step=1e-3):
    """Estimate a parameter-space directional second derivative by central difference.

    The probe evaluates ``loss_fn`` at ``theta``, ``theta + step * direction`` and
    ``theta - step * direction`` under ``no_grad()``. All three evaluations replay
    the same NumPy global RNG state. Parameter values, bindings, gradients, and
    caller RNG state are restored before returning. Tensor mutation versions are
    intentionally left monotonic so graphs built before a nonzero probe remain stale.
    """
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    step = _validate_step(step)

    with _PROBE_LOCK:
        parameters = _materialize_parameters(parameters)
        directions = _materialize_direction(direction, parameters)
        entries = _snapshot_entries(parameters)
        plus_values, minus_values = _candidate_values(entries, directions, step)

        _preflight_install(parameters, plus_values)
        _preflight_install(parameters, minus_values)

        rng_state = _snapshot_rng_state()
        try:
            baseline_values = tuple(entry["data"] for entry in entries)
            baseline = _evaluate(
                loss_fn, parameters, entries, baseline_values, rng_state
            )

            _install(parameters, plus_values)
            plus = _evaluate(loss_fn, parameters, entries, plus_values, rng_state)
            _install(parameters, baseline_values)

            _install(parameters, minus_values)
            minus = _evaluate(loss_fn, parameters, entries, minus_values, rng_state)
            _install(parameters, baseline_values)

            curvature, overflow, underflow, sign = _curvature_value(
                baseline, plus, minus, step
            )
            changed_parameters = sum(
                not np.array_equal(plus_value, entry["data"])
                or not np.array_equal(minus_value, entry["data"])
                for plus_value, minus_value, entry in zip(
                    plus_values, minus_values, entries
                )
            )
            changed_elements = sum(
                int(
                    np.count_nonzero(plus_value != entry["data"])
                    + np.count_nonzero(minus_value != entry["data"])
                )
                for plus_value, minus_value, entry in zip(
                    plus_values, minus_values, entries
                )
            )
            return {
                "baseline_loss": baseline,
                "plus_loss": plus,
                "minus_loss": minus,
                "step": step,
                "curvature": curvature,
                "curvature_overflow": overflow,
                "curvature_underflow": underflow,
                "curvature_sign": sign,
                "parameter_count": len(parameters),
                "changed_parameter_count": int(changed_parameters),
                "changed_perturbation_elements": int(changed_elements),
                "method": "central_finite_difference",
            }
        finally:
            try:
                _restore_entries(parameters, entries)
            finally:
                np.random.set_state(rng_state)
