"""Unitwise adaptive gradient clipping with overflow-stable norm comparisons."""

from fractions import Fraction
import math
import threading
import weakref

import numpy as np

from .tensor import Tensor, _VersionedArray


_AGC_LOCK = threading.RLock()
_ZERO_SCALED = (0.0, 0)


def _positive_real(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, np.floating):
        source = np.asarray(value)
        dtype = source.dtype
        if not bool(np.isfinite(source)):
            raise ValueError(f"{name} must be finite")
        if dtype.itemsize > np.dtype(np.float64).itemsize:
            limit = np.array(np.finfo(np.float64).max, dtype=dtype)
            with np.errstate(over="ignore", invalid="raise", under="ignore"):
                if bool(np.abs(source) > limit):
                    raise ValueError(f"{name} must fit float64")
        value = source[()]
    elif isinstance(value, np.integer):
        value = np.asarray(value)[()]
    elif isinstance(value, float):
        value = float.real.__get__(value, float)
    elif isinstance(value, int):
        value = int.real.__get__(value, int)
    elif type(value) is not Fraction:
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
        if type(parameter) is not Tensor:
            raise TypeError(f"parameter {index} must be a Tensor")
        children = parameter._children
        if type(children) is not tuple:
            raise TypeError(f"parameter {index} graph metadata must be a plain tuple")
        if children != ():
            raise ValueError(f"parameter {index} must be a leaf Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(marker)
    return materialized


def _float64_copy(array, *, name, shape, floating_only):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    base = np.asarray(array)
    if base.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {base.shape}")
    if floating_only:
        if not np.issubdtype(base.dtype, np.floating):
            raise TypeError(f"{name} must have a floating dtype")
    elif (
        not np.issubdtype(base.dtype, np.number)
        or np.issubdtype(base.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.all(np.isfinite(base)):
        raise ValueError(f"{name} must contain only finite values")

    if base.dtype.itemsize > np.dtype(np.float64).itemsize:
        limit = np.array(np.finfo(np.float64).max, dtype=base.dtype)
        with np.errstate(over="ignore", invalid="raise", under="ignore"):
            if np.any(np.abs(base) > limit):
                raise ValueError(f"{name} must fit float64")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.asarray(base, dtype=np.float64).copy()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must fit float64")
    return result


def _array_equal(left, right):
    """Compare ndarray storage without trusting subclass NumPy dispatch hooks."""

    left_base = np.asarray(left)
    right_base = np.asarray(right)
    return bool(np.array_equal(left_base, right_base))


def _is_writable(array):
    """Read storage metadata without trusting ndarray subclass attribute overrides."""

    return bool(np.asarray(array).flags.writeable)


def _is_tensor_managed_storage(array, parameter):
    """Reject malformed or foreign Tensor storage ownership metadata."""

    if type(array) is not _VersionedArray:
        return True
    owner_ref = array._owner_ref
    if type(owner_ref) is not weakref.ReferenceType:
        return False
    return owner_ref() is parameter


def _storage_owner_ref(array):
    """Snapshot validated Tensor-storage ownership metadata without dereferencing it."""

    if type(array) is not _VersionedArray:
        return None
    return array._owner_ref


def _has_expected_storage_owner(array, parameter, expected_owner_ref):
    """Check storage ownership identity without trusting mutable callable metadata."""

    if type(array) is not _VersionedArray:
        return expected_owner_ref is None
    owner_ref = array._owner_ref
    if type(owner_ref) is not weakref.ReferenceType or owner_ref is not expected_owner_ref:
        return False
    return owner_ref() is parameter


def _is_plain_shape(shape):
    """Recognize shape metadata without invoking integer-subclass comparisons."""

    return type(shape) is tuple and all(type(dimension) is int for dimension in shape)


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


def _native_candidate(clipped, gradient_dtype, target_norm):
    """Round to the live dtype without letting rounding exceed the AGC bound."""

    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            native = np.asarray(clipped, dtype=gradient_dtype)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("adaptive clipping candidate must fit gradient dtype") from exc

    native_norm, _, _ = _norm_descriptor(np.asarray(native, dtype=np.float64))
    if _scaled_compare(native_norm, target_norm) <= 0:
        return native

    # Nearest rounding can move a low-precision component away from zero. One
    # representable step toward zero puts every component at or inside the exact
    # float64 candidate, so the stored vector cannot remain outside its norm bound.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        native = np.nextafter(native, np.zeros((), dtype=gradient_dtype))
    native_norm, _, _ = _norm_descriptor(np.asarray(native, dtype=np.float64))
    if _scaled_compare(native_norm, target_norm) > 0:
        raise ValueError("adaptive clipping candidate cannot satisfy gradient dtype bound")
    return native


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
        native = _native_candidate(clipped, gradient_dtype, target_norm)
        if not _array_equal(candidate_rows[index], native):
            candidate_rows[index][...] = native
            clipped_units += 1

    return candidate, clipped_units


def _shares_memory(left, right, label):
    """Check exact overlap without trusting ndarray subclass dispatch hooks."""

    left_base = np.asarray(left)
    right_base = np.asarray(right)
    try:
        return bool(np.shares_memory(left_base, right_base))
    except ValueError as exc:
        raise ValueError(f"{label} storage overlap could not be determined") from exc


def _preflight(destinations, candidates, parameter_data):
    changed = [
        index
        for index, (destination, candidate) in enumerate(zip(destinations, candidates))
        if destination is not None and not _array_equal(destination, candidate)
    ]

    for index in changed:
        if not _is_writable(destinations[index]):
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


def _validate_transaction_metadata(
    parameters,
    destinations,
    expected_gradients,
    trainability,
    grad_shapes,
    data_bindings,
    data_values,
    data_versions,
    data_owner_refs,
):
    for index, (
        parameter,
        destination,
        expected_trainable,
        expected_grad_shape,
        expected_data,
        expected_values,
        expected_version,
        expected_owner_ref,
    ) in enumerate(
        zip(
            parameters,
            destinations,
            trainability,
            grad_shapes,
            data_bindings,
            data_values,
            data_versions,
            data_owner_refs,
        )
    ):
        if parameter.grad is not destination:
            raise RuntimeError(
                f"adaptive gradient clipping gradient binding changed for parameter {index}"
            )
        expected_gradient = expected_gradients[index]
        if destination is not None and not _array_equal(destination, expected_gradient):
            raise RuntimeError(
                f"adaptive gradient clipping gradient value changed for parameter {index}"
            )
        current_grad_shape = parameter._grad_shape
        if not _is_plain_shape(current_grad_shape) or current_grad_shape != expected_grad_shape:
            raise RuntimeError(
                "adaptive gradient clipping gradient shape metadata changed for parameter "
                f"{index}"
            )
        children = parameter._children
        if type(children) is not tuple or children != ():
            raise RuntimeError(
                f"adaptive gradient clipping graph metadata changed for parameter {index}"
            )
        if parameter.requires_grad is not expected_trainable:
            raise RuntimeError(
                f"adaptive gradient clipping trainability changed for parameter {index}"
            )
        if parameter.data is not expected_data:
            raise RuntimeError(
                f"adaptive gradient clipping parameter data binding changed for parameter {index}"
            )
        if not _has_expected_storage_owner(
            parameter.data, parameter, expected_owner_ref
        ):
            raise RuntimeError(
                "adaptive gradient clipping parameter storage ownership changed for parameter "
                f"{index}"
            )
        if not _array_equal(parameter.data, expected_values):
            raise RuntimeError(
                f"adaptive gradient clipping parameter data changed for parameter {index}"
            )
        current_version = parameter._version
        if type(current_version) is not int or current_version != expected_version:
            raise RuntimeError(
                f"adaptive gradient clipping parameter version changed for parameter {index}"
            )


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
        data_values = []
        data_versions = []
        data_owner_refs = []
        destinations = []
        originals = []
        candidates = []
        trainability = []
        grad_shapes = []
        total_clipped_units = 0

        for index, parameter in enumerate(parameters):
            requires_grad = parameter.requires_grad
            if not isinstance(requires_grad, bool):
                raise TypeError(f"parameter {index} requires_grad must be a bool")
            trainability.append(requires_grad)

            version = parameter._version
            if type(version) is not int:
                raise TypeError(f"parameter {index} version must be an int")
            if version < 0:
                raise ValueError(f"parameter {index} version must be non-negative")

            data = parameter.data
            if not isinstance(data, np.ndarray):
                raise TypeError(f"parameter {index} data must be a NumPy array")
            if not _is_tensor_managed_storage(data, parameter):
                raise TypeError(f"parameter {index} data must be Tensor-managed storage")
            data_owner_refs.append(_storage_owner_ref(data))
            data_shape = np.asarray(data).shape
            grad_shape = parameter._grad_shape
            if type(grad_shape) is not tuple:
                raise TypeError(
                    f"parameter {index} gradient shape metadata must be a plain tuple"
                )
            if not _is_plain_shape(grad_shape):
                raise TypeError(
                    f"parameter {index} gradient shape metadata dimensions must be plain ints"
                )
            if grad_shape != data_shape:
                raise ValueError(
                    f"parameter {index} gradient shape metadata must match data shape"
                )
            grad_shapes.append(grad_shape)

            data_snapshot = _float64_copy(
                data,
                name=f"parameter {index} data",
                shape=data_shape,
                floating_only=False,
            )
            parameter_data.append(data)
            data_values.append(np.array(data, copy=True))
            data_versions.append(version)

            gradient = parameter.grad
            if gradient is None:
                destinations.append(None)
                originals.append(None)
                candidates.append(None)
                continue
            if not requires_grad:
                raise ValueError(f"parameter {index} is frozen but still has a gradient")

            gradient_snapshot = _float64_copy(
                gradient,
                name=f"gradient for parameter {index}",
                shape=data_shape,
                floating_only=True,
            )
            candidate, clipped_units = _candidate_gradient(
                data_snapshot,
                gradient_snapshot,
                np.asarray(gradient).dtype,
                clip_factor,
                eps,
            )
            destinations.append(gradient)
            originals.append(np.array(gradient, copy=True))
            candidates.append(candidate)
            total_clipped_units += clipped_units

        changed = _preflight(destinations, candidates, parameter_data)
        expected_gradients = list(originals)
        try:
            for index in changed:
                # Caller-controlled ndarray __setitem__ receives the assigned object and
                # may mutate it. Keep the canonical candidate private for postvalidation.
                write_candidate = np.array(candidates[index], copy=True)
                destinations[index][...] = write_candidate
                if not _array_equal(destinations[index], candidates[index]):
                    raise RuntimeError(
                        f"adaptive gradient clipping write failed for parameter {index}"
                    )
                expected_gradients[index] = candidates[index]
                _validate_transaction_metadata(
                    parameters,
                    destinations,
                    expected_gradients,
                    trainability,
                    grad_shapes,
                    parameter_data,
                    data_values,
                    data_versions,
                    data_owner_refs,
                )
        except BaseException:
            rollback_error = None
            for index in reversed(range(len(destinations))):
                destination = destinations[index]
                original = originals[index]
                if destination is None:
                    continue
                try:
                    if _array_equal(destination, original):
                        continue
                    # Keep the canonical entry snapshot private from caller-controlled
                    # rollback writes, just as commit keeps its candidate private.
                    write_original = np.array(original, copy=True)
                    destination[...] = write_original
                    if not _array_equal(destination, original):
                        raise RuntimeError("rollback postcondition failed")
                except BaseException as exc:  # best-effort cleanup across every gradient
                    if rollback_error is None:
                        rollback_error = exc

            # Parameter-data restoration may execute ndarray-subclass __setitem__.
            # Do it before metadata repair so no caller hook runs after the final
            # transaction metadata postconditions have been re-established.
            for parameter, expected_data, expected_values, expected_owner_ref in zip(
                parameters, parameter_data, data_values, data_owner_refs
            ):
                try:
                    if parameter.data is not expected_data:
                        parameter._data = expected_data
                    if type(expected_data) is _VersionedArray:
                        if expected_data._owner_ref is not expected_owner_ref:
                            expected_data._owner_ref = expected_owner_ref
                        if not _has_expected_storage_owner(
                            expected_data, parameter, expected_owner_ref
                        ):
                            raise RuntimeError(
                                "parameter storage ownership rollback postcondition failed"
                            )
                    if not _array_equal(expected_data, expected_values):
                        # Keep the canonical parameter entry snapshot private from
                        # caller-controlled storage assignment hooks.
                        write_values = np.array(expected_values, copy=True)
                        expected_data[...] = write_values
                    # A hostile ndarray write may also rebind storage or corrupt
                    # Tensor-managed ownership, so repair those after the write.
                    if parameter.data is not expected_data:
                        parameter._data = expected_data
                    if type(expected_data) is _VersionedArray:
                        if expected_data._owner_ref is not expected_owner_ref:
                            expected_data._owner_ref = expected_owner_ref
                        if not _has_expected_storage_owner(
                            expected_data, parameter, expected_owner_ref
                        ):
                            raise RuntimeError(
                                "parameter storage ownership rollback postcondition failed"
                            )
                    if parameter.data is not expected_data:
                        raise RuntimeError("parameter data binding rollback failed")
                    if not _array_equal(parameter.data, expected_values):
                        raise RuntimeError("parameter data rollback postcondition failed")
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            for parameter, destination in zip(parameters, destinations):
                try:
                    if parameter.grad is not destination:
                        parameter.grad = destination
                    if parameter.grad is not destination:
                        raise RuntimeError("gradient binding rollback postcondition failed")
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            for parameter, expected_grad_shape in zip(parameters, grad_shapes):
                try:
                    if (
                        not _is_plain_shape(parameter._grad_shape)
                        or parameter._grad_shape != expected_grad_shape
                    ):
                        parameter._grad_shape = expected_grad_shape
                    if (
                        not _is_plain_shape(parameter._grad_shape)
                        or parameter._grad_shape != expected_grad_shape
                    ):
                        raise RuntimeError(
                            "gradient shape metadata rollback postcondition failed"
                        )
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            for parameter in parameters:
                try:
                    children = parameter._children
                    if type(children) is not tuple or children != ():
                        parameter._children = ()
                    children = parameter._children
                    if type(children) is not tuple or children != ():
                        raise RuntimeError("graph metadata rollback postcondition failed")
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            for parameter, expected_trainable in zip(parameters, trainability):
                try:
                    if parameter.requires_grad is not expected_trainable:
                        parameter.requires_grad = expected_trainable
                    if parameter.requires_grad is not expected_trainable:
                        raise RuntimeError("trainability rollback postcondition failed")
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            for parameter, expected_version in zip(parameters, data_versions):
                try:
                    current_version = parameter._version
                    if type(current_version) is not int or current_version < expected_version:
                        parameter._version = expected_version
                        current_version = parameter._version
                    if type(current_version) is not int or current_version < expected_version:
                        raise RuntimeError("parameter version rollback postcondition failed")
                except BaseException as exc:
                    if rollback_error is None:
                        rollback_error = exc

            if rollback_error is not None:
                raise RuntimeError("adaptive gradient clipping rollback failed") from rollback_error
            raise

        _validate_transaction_metadata(
            parameters,
            destinations,
            candidates,
            trainability,
            grad_shapes,
            parameter_data,
            data_values,
            data_versions,
            data_owner_refs,
        )
        return total_clipped_units