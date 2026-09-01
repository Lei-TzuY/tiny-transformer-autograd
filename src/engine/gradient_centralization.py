"""In-place gradient centralization with explicit transactional validation."""

import numbers
import warnings

import numpy as np

from .tensor import Tensor


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        items = (parameters,)
    else:
        try:
            items = tuple(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be a Tensor or iterable of Tensors") from exc

    seen = set()
    for index, parameter in enumerate(items):
        if type(parameter) is not Tensor:
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor identities")
        seen.add(marker)
    return items


def _validate_min_rank(min_rank):
    is_bool = isinstance(min_rank, (bool, np.bool_))
    if is_bool or not isinstance(min_rank, numbers.Integral):
        raise TypeError("min_rank must be an integer")
    value = int(min_rank)
    if value < 2:
        raise ValueError("min_rank must be at least 2")
    return value


def _shares_memory(left, right):
    try:
        return bool(np.shares_memory(np.asarray(left), np.asarray(right)))
    except ValueError as exc:
        raise ValueError("gradient storage overlap could not be determined") from exc


def _restore_array_dtype(array, expected_dtype):
    if np.asarray(array).dtype == expected_dtype:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.dtype.__set__(array, expected_dtype)


def _restore_array_shape(array, expected_shape):
    if np.asarray(array).shape == expected_shape:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.shape.__set__(array, expected_shape)


def _restore_array_strides(array, expected_strides):
    if np.asarray(array).strides == expected_strides:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.strides.__set__(array, expected_strides)


def _restore_array_writeable(array, expected_writeable):
    if bool(np.asarray(array).flags.writeable) == expected_writeable:
        return
    np.ndarray.setflags(array, write=expected_writeable)


def _centralized_candidate(gradient):
    source = np.asarray(gradient)
    shape = source.shape
    trailing_size = int(np.prod(shape[1:], dtype=np.intp))
    rows = source.reshape((shape[0], trailing_size))
    candidate = np.array(source, dtype=source.dtype, copy=True)
    candidate_rows = candidate.reshape(rows.shape)

    for index, row in enumerate(rows):
        scale = float(np.max(np.abs(row))) if row.size else 0.0
        if scale == 0.0:
            continue
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"
            ):
                normalized = np.asarray(row, dtype=np.float64) / scale
                mean = float(np.sum(normalized) / normalized.size) * scale
                centered = np.asarray(row, dtype=np.float64) - mean
        except FloatingPointError as exc:
            raise ValueError(
                "centralized gradient is not representable in float64"
            ) from exc
        if not np.all(np.isfinite(centered)):
            raise ValueError("centralized gradient is not representable in float64")
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                native = np.asarray(centered, dtype=source.dtype)
        except FloatingPointError as exc:
            raise ValueError("centralized gradient does not fit its original dtype") from exc
        if not np.all(np.isfinite(native)):
            raise ValueError("centralized gradient does not fit its original dtype")
        candidate_rows[index][...] = native

    return candidate


def _validate_transaction_state(
    parameters,
    gradients,
    originals,
    entry_dtypes,
    entry_strides,
    entry_writeable,
    candidates,
    committed,
):
    for index, gradient in enumerate(gradients):
        if originals[index] is None:
            continue
        if parameters[index].grad is not gradient:
            raise RuntimeError(
                f"gradient binding changed for parameter {index} during centralization"
            )
        if np.asarray(gradient).shape != originals[index].shape:
            raise RuntimeError(
                f"gradient shape changed for parameter {index} during centralization"
            )
        if np.asarray(gradient).dtype != entry_dtypes[index]:
            raise RuntimeError(
                f"gradient dtype changed for parameter {index} during centralization"
            )
        if np.asarray(gradient).strides != entry_strides[index]:
            raise RuntimeError(
                f"gradient strides changed for parameter {index} during centralization"
            )
        if bool(np.asarray(gradient).flags.writeable) != entry_writeable[index]:
            raise RuntimeError(
                f"gradient writability changed for parameter {index} during centralization"
            )
        expected = candidates[index] if index in committed else originals[index]
        if not np.array_equal(np.asarray(gradient), expected):
            raise RuntimeError(
                f"gradient value changed for parameter {index} during centralization"
            )


def centralize_gradients_(parameters, *, min_rank=2):
    """Center eligible live gradients across every non-leading axis in-place."""

    min_rank = _validate_min_rank(min_rank)
    parameters = _materialize_parameters(parameters)
    gradients = []
    originals = []
    entry_dtypes = []
    entry_strides = []
    entry_writeable = []
    candidates = []
    changed = []

    for index, parameter in enumerate(parameters):
        requires_grad = parameter.requires_grad
        if not isinstance(requires_grad, bool):
            raise TypeError(f"parameter {index} requires_grad must be a bool")

        gradient = parameter.grad
        gradients.append(gradient)
        if gradient is None:
            originals.append(None)
            entry_dtypes.append(None)
            entry_strides.append(None)
            entry_writeable.append(None)
            candidates.append(None)
            continue
        if not requires_grad:
            raise ValueError(f"parameter {index} is frozen but still has a gradient")
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"gradient for parameter {index} must be a NumPy array")

        base = np.asarray(gradient)
        parameter_data = np.asarray(parameter.data)
        expected_shape = parameter_data.shape
        if base.shape != expected_shape:
            raise ValueError(
                f"gradient for parameter {index} shape mismatch: "
                f"expected {expected_shape}, got {base.shape}"
            )
        if not np.issubdtype(base.dtype, np.floating):
            raise TypeError(f"gradient for parameter {index} must have a floating dtype")
        if not np.all(np.isfinite(base)):
            raise ValueError(
                f"gradient for parameter {index} must contain only finite values"
            )
        if parameter_data.ndim < min_rank:
            originals.append(None)
            entry_dtypes.append(None)
            entry_strides.append(None)
            entry_writeable.append(None)
            candidates.append(None)
            continue

        original = np.array(base, copy=True)
        candidate = _centralized_candidate(base)
        originals.append(original)
        entry_dtypes.append(base.dtype)
        entry_strides.append(base.strides)
        entry_writeable.append(bool(base.flags.writeable))
        candidates.append(candidate)
        if not np.array_equal(base, candidate):
            if not bool(base.flags.writeable):
                raise ValueError(f"gradient for parameter {index} must be writable")
            changed.append(index)

    for gradient_index in changed:
        for parameter_index, parameter in enumerate(parameters):
            if _shares_memory(gradients[gradient_index], parameter.data):
                raise ValueError(
                    f"gradient for parameter {gradient_index} must not overlap "
                    f"parameter {parameter_index} data"
                )

    for right_position, right_index in enumerate(changed):
        for left_index in changed[:right_position]:
            if _shares_memory(gradients[left_index], gradients[right_index]):
                raise ValueError(
                    "gradient storage must not overlap between parameters "
                    f"{left_index} and {right_index}"
                )

    attempted = []
    committed = set()
    try:
        for index in changed:
            destination = gradients[index]
            if parameters[index].grad is not destination:
                raise RuntimeError(
                    f"gradient binding changed for parameter {index} before centralization"
                )
            attempted.append(index)
            destination[...] = np.array(candidates[index], copy=True)
            if np.asarray(destination).shape != originals[index].shape:
                raise RuntimeError(
                    f"gradient centralization write failed for parameter {index}"
                )
            if np.asarray(destination).dtype != entry_dtypes[index]:
                raise RuntimeError(
                    f"gradient dtype changed for parameter {index} during centralization"
                )
            if np.asarray(destination).strides != entry_strides[index]:
                raise RuntimeError(
                    f"gradient strides changed for parameter {index} during centralization"
                )
            if bool(np.asarray(destination).flags.writeable) != entry_writeable[index]:
                raise RuntimeError(
                    f"gradient writability changed for parameter {index} "
                    "during centralization"
                )
            if not np.array_equal(np.asarray(destination), candidates[index]):
                raise RuntimeError(
                    f"gradient centralization write failed for parameter {index}"
                )
            committed.add(index)
            _validate_transaction_state(
                parameters,
                gradients,
                originals,
                entry_dtypes,
                entry_strides,
                entry_writeable,
                candidates,
                committed,
            )
    except BaseException:
        rollback_error = None

        for index, gradient in enumerate(gradients):
            try:
                if parameters[index].grad is not gradient:
                    parameters[index].grad = gradient
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc

        for index in reversed(range(len(gradients))):
            if originals[index] is None:
                continue
            try:
                destination = gradients[index]
                _restore_array_dtype(destination, entry_dtypes[index])
                _restore_array_shape(destination, originals[index].shape)
                _restore_array_strides(destination, entry_strides[index])
                needs_value_repair = not np.array_equal(
                    np.asarray(destination), originals[index]
                )
                if needs_value_repair and not bool(
                    np.asarray(destination).flags.writeable
                ):
                    np.ndarray.setflags(destination, write=True)
                if needs_value_repair:
                    np.ndarray.__setitem__(
                        destination, Ellipsis, np.array(originals[index], copy=True)
                    )
                    if not np.array_equal(np.asarray(destination), originals[index]):
                        raise RuntimeError(
                            "gradient centralization rollback postcondition failed"
                        )
                _restore_array_writeable(destination, entry_writeable[index])
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc

        for index, gradient in enumerate(gradients):
            try:
                if parameters[index].grad is not gradient:
                    raise RuntimeError(
                        "gradient centralization binding rollback postcondition failed"
                    )
                if originals[index] is None:
                    continue
                if np.asarray(gradient).shape != originals[index].shape:
                    raise RuntimeError(
                        "gradient centralization shape rollback postcondition failed"
                    )
                if np.asarray(gradient).dtype != entry_dtypes[index]:
                    raise RuntimeError(
                        "gradient centralization dtype rollback postcondition failed"
                    )
                if np.asarray(gradient).strides != entry_strides[index]:
                    raise RuntimeError(
                        "gradient centralization strides rollback postcondition failed"
                    )
                if bool(np.asarray(gradient).flags.writeable) != entry_writeable[index]:
                    raise RuntimeError(
                        "gradient centralization writability rollback postcondition failed"
                    )
                if not np.array_equal(np.asarray(gradient), originals[index]):
                    raise RuntimeError(
                        "gradient centralization value rollback postcondition failed"
                    )
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc

        if rollback_error is not None:
            raise RuntimeError("gradient centralization rollback failed") from rollback_error
        raise

    return len(changed)


import threading as _threading

_CENTRALIZATION_LOCK = _threading.RLock()
_centralize_gradients_unlocked = centralize_gradients_


def _snapshot_parameter_guard_state(parameters):
    states = []
    for parameter in parameters:
        data = parameter.data
        data_base = np.asarray(data)
        gradient = parameter.grad
        gradient_state = None
        if isinstance(gradient, np.ndarray):
            gradient_base = np.asarray(gradient)
            gradient_state = (
                gradient,
                np.array(gradient_base, copy=True),
                gradient_base.dtype,
                gradient_base.strides,
                bool(gradient_base.flags.writeable),
            )
        states.append(
            (
                data,
                np.array(data_base, copy=True),
                data_base.dtype,
                data_base.strides,
                bool(data_base.flags.writeable),
                parameter._version,
                gradient,
                gradient_state,
            )
        )
    return tuple(states)


def _validate_parameter_guard_state(parameters, states):
    for index, (parameter, state) in enumerate(zip(parameters, states)):
        data, values, dtype, strides, writeable, version, _, _ = state
        current = parameter.data
        if current is not data:
            raise RuntimeError(
                f"parameter data changed for parameter {index} during centralization"
            )
        base = np.asarray(current)
        if base.shape != values.shape or base.dtype != dtype or base.strides != strides:
            raise RuntimeError(
                f"parameter data changed for parameter {index} during centralization"
            )
        if bool(base.flags.writeable) != writeable:
            raise RuntimeError(
                f"parameter data changed for parameter {index} during centralization"
            )
        if parameter._version != version or not np.array_equal(base, values):
            raise RuntimeError(
                f"parameter data changed for parameter {index} during centralization"
            )


def _restore_parameter_guard_state(parameters, states, *, restore_gradients):
    rollback_error = None
    for parameter, state in zip(parameters, states):
        data, values, dtype, strides, writeable, _, gradient, gradient_state = state
        try:
            if parameter.data is not data:
                parameter._data = data
            _restore_array_dtype(data, dtype)
            _restore_array_shape(data, values.shape)
            _restore_array_strides(data, strides)
            if not bool(np.asarray(data).flags.writeable):
                np.ndarray.setflags(data, write=True)
            if not np.array_equal(np.asarray(data), values):
                np.ndarray.__setitem__(data, Ellipsis, np.array(values, copy=True))
            _restore_array_writeable(data, writeable)

            if restore_gradients:
                if parameter.grad is not gradient:
                    parameter.grad = gradient
                if gradient_state is not None:
                    grad, grad_values, grad_dtype, grad_strides, grad_writeable = gradient_state
                    _restore_array_dtype(grad, grad_dtype)
                    _restore_array_shape(grad, grad_values.shape)
                    _restore_array_strides(grad, grad_strides)
                    if not bool(np.asarray(grad).flags.writeable):
                        np.ndarray.setflags(grad, write=True)
                    if not np.array_equal(np.asarray(grad), grad_values):
                        np.ndarray.__setitem__(
                            grad, Ellipsis, np.array(grad_values, copy=True)
                        )
                    _restore_array_writeable(grad, grad_writeable)
        except BaseException as exc:
            if rollback_error is None:
                rollback_error = exc
    if rollback_error is not None:
        raise RuntimeError("gradient centralization outer rollback failed") from rollback_error


_centralize_gradients_with_gradient_guards = centralize_gradients_


def centralize_gradients_(parameters, *, min_rank=2):
    """Center eligible gradients without allowing parameter-state side effects."""

    with _CENTRALIZATION_LOCK:
        materialized = _materialize_parameters(parameters)
        states = _snapshot_parameter_guard_state(materialized)
        inner_succeeded = False
        try:
            changed = _centralize_gradients_unlocked(materialized, min_rank=min_rank)
            inner_succeeded = True
            _validate_parameter_guard_state(materialized, states)
            return changed
        except BaseException:
            _restore_parameter_guard_state(
                materialized, states, restore_gradients=inner_succeeded
            )
            raise
