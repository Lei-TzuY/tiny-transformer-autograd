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
    # Dtype controls how the same bytes are interpreted, so repair it before shape,
    # strides, and values. Use the ndarray descriptor to avoid subclass hooks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.dtype.__set__(array, expected_dtype)


def _restore_array_shape(array, expected_shape):
    if np.asarray(array).shape == expected_shape:
        return
    # Exceptional rollback must preserve the exact caller-owned ndarray object.
    # New NumPy versions deprecate direct shape assignment, so suppress only that
    # deprecation at this narrow repair boundary rather than weakening -W error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np.ndarray.shape.__set__(array, expected_shape)


def _restore_array_strides(array, expected_strides):
    if np.asarray(array).strides == expected_strides:
        return
    # Strides are caller-visible storage metadata. Repair them before writing entry
    # values so a zero-stride alias cannot make rollback overwrite only one slot.
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
    """Center eligible live gradients across every non-leading axis in-place.

    For a rank-2 Linear-style weight gradient ``(out, in)``, each output row is
    centered independently. Higher-rank tensors are treated the same way: axis 0
    identifies units and all remaining axes form that unit. Gradients below
    ``min_rank`` and missing gradients are left unchanged.

    The complete parameter/gradient collection is validated and every candidate is
    computed before the first write. The function returns the number of gradient
    tensors whose stored values changed.
    """

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
        if np.asarray(parameter.data).ndim < min_rank:
            originals.append(None)
            entry_dtypes.append(None)
            entry_strides.append(None)
            entry_writeable.append(None)
            candidates.append(None)
            continue
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"gradient for parameter {index} must be a NumPy array")

        base = np.asarray(gradient)
        expected_shape = np.asarray(parameter.data).shape
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

    # Gradient centralization owns gradient buffers only. A write into storage that
    # aliases any bound parameter would silently mutate model weights and may advance
    # Tensor mutation versions. Keep no-op aliases valid because they perform no write.
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

        # A caller-controlled ndarray write can mutate another Tensor's public
        # ``grad`` attribute even when that other destination has not been reached
        # yet. Restore the complete collection's entry bindings before repairing
        # storage values so a failed transaction cannot leak rebinding.
        for index, gradient in enumerate(gradients):
            try:
                if parameters[index].grad is not gradient:
                    parameters[index].grad = gradient
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc

        # A write hook can mutate another eligible gradient's metadata/value. Dtype
        # must be repaired before shape/strides because it controls byte interpretation.
        # Writability is temporarily enabled only when value repair needs it.
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


# A transaction can execute caller-controlled ndarray subclass hooks while its entry
# snapshots are live. Serialize complete helper-managed calls so another transaction
# cannot observe or mutate a half-committed collection. RLock keeps same-thread nested
# diagnostics on disjoint parameters usable instead of introducing a self-deadlock.
import threading as _threading

_CENTRALIZATION_LOCK = _threading.RLock()
_centralize_gradients_unlocked = centralize_gradients_


def centralize_gradients_(parameters, *, min_rank=2):
    """Center eligible gradients atomically across helper-managed concurrent calls."""

    with _CENTRALIZATION_LOCK:
        return _centralize_gradients_unlocked(parameters, min_rank=min_rank)
