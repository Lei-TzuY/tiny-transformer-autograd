"""Public gradient-centralization guard for complete parameter/gradient metadata."""

import numpy as np

from .gradient_centralization import (
    _CENTRALIZATION_LOCK,
    _materialize_parameters,
    _restore_array_dtype,
    _restore_array_shape,
    _restore_array_strides,
    _restore_array_writeable,
    centralize_gradients_ as _centralize_gradients_impl,
)


def _snapshot_gradients(parameters):
    snapshots = []
    for parameter in parameters:
        gradient = parameter.grad
        if isinstance(gradient, np.ndarray):
            base = np.asarray(gradient)
            snapshots.append(
                (
                    gradient,
                    np.array(base, copy=True),
                    base.dtype,
                    base.strides,
                    bool(base.flags.writeable),
                )
            )
        else:
            snapshots.append((gradient, None, None, None, None))
    return tuple(snapshots)


def _validate_gradients(parameters, gradients):
    for index, (parameter, gradient_state) in enumerate(zip(parameters, gradients)):
        gradient, values, dtype, strides, writeable = gradient_state
        if parameter.grad is not gradient:
            raise RuntimeError(
                f"gradient binding changed for parameter {index} during centralization"
            )
        if values is None:
            continue
        base = np.asarray(gradient)
        if base.shape != values.shape:
            raise RuntimeError(
                f"gradient shape changed for parameter {index} during centralization"
            )
        if base.dtype != dtype:
            raise RuntimeError(
                f"gradient dtype changed for parameter {index} during centralization"
            )
        if base.strides != strides:
            raise RuntimeError(
                f"gradient strides changed for parameter {index} during centralization"
            )
        if bool(base.flags.writeable) != writeable:
            raise RuntimeError(
                f"gradient writability changed for parameter {index} during centralization"
            )
        if not np.array_equal(base, values):
            raise RuntimeError(
                f"gradient value changed for parameter {index} during centralization"
            )


def _restore_trainability_and_gradients(parameters, trainability, gradients):
    rollback_error = None
    for parameter, requires_grad, gradient_state in zip(
        parameters, trainability, gradients
    ):
        gradient, values, dtype, strides, writeable = gradient_state
        try:
            parameter.requires_grad = requires_grad
            if parameter.grad is not gradient:
                parameter.grad = gradient
            if values is None:
                continue
            _restore_array_dtype(gradient, dtype)
            _restore_array_shape(gradient, values.shape)
            _restore_array_strides(gradient, strides)
            if not bool(np.asarray(gradient).flags.writeable):
                np.ndarray.setflags(gradient, write=True)
            if not np.array_equal(np.asarray(gradient), values):
                np.ndarray.__setitem__(
                    gradient, Ellipsis, np.array(values, copy=True)
                )
            _restore_array_writeable(gradient, writeable)
        except BaseException as exc:
            if rollback_error is None:
                rollback_error = exc
    if rollback_error is not None:
        raise RuntimeError("gradient centralization public rollback failed") from rollback_error


def centralize_gradients_(parameters, *, min_rank=2):
    """Center gradients transactionally while preserving all entry gradient state."""

    with _CENTRALIZATION_LOCK:
        materialized = _materialize_parameters(parameters)
        trainability = tuple(parameter.requires_grad for parameter in materialized)
        gradients = _snapshot_gradients(materialized)
        try:
            changed = _centralize_gradients_impl(materialized, min_rank=min_rank)
            for index, (parameter, entry) in enumerate(
                zip(materialized, trainability)
            ):
                if parameter.requires_grad is not entry:
                    raise RuntimeError(
                        f"parameter trainability changed for parameter {index} "
                        "during centralization"
                    )
            _validate_gradients(materialized, gradients)
            return changed
        except BaseException:
            _restore_trainability_and_gradients(
                materialized, trainability, gradients
            )
            raise
