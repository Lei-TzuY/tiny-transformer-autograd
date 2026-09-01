"""Public gradient-centralization guard for Tensor trainability metadata."""

import numpy as np

from .gradient_centralization import (
    _CENTRALIZATION_LOCK,
    _materialize_parameters,
    centralize_gradients_ as _centralize_gradients_impl,
)


def _snapshot_gradients(parameters):
    snapshots = []
    for parameter in parameters:
        gradient = parameter.grad
        values = None
        if isinstance(gradient, np.ndarray):
            values = np.array(np.asarray(gradient), copy=True)
        snapshots.append((gradient, values))
    return tuple(snapshots)


def _restore_trainability_and_gradients(parameters, trainability, gradients):
    rollback_error = None
    for parameter, requires_grad, gradient_state in zip(
        parameters, trainability, gradients
    ):
        gradient, values = gradient_state
        try:
            parameter.requires_grad = requires_grad
            if parameter.grad is not gradient:
                parameter.grad = gradient
            if values is not None and not np.array_equal(np.asarray(gradient), values):
                writable = bool(np.asarray(gradient).flags.writeable)
                if not writable:
                    np.ndarray.setflags(gradient, write=True)
                np.ndarray.__setitem__(
                    gradient, Ellipsis, np.array(values, copy=True)
                )
                if not writable:
                    np.ndarray.setflags(gradient, write=False)
        except BaseException as exc:
            if rollback_error is None:
                rollback_error = exc
    if rollback_error is not None:
        raise RuntimeError("gradient centralization trainability rollback failed") from rollback_error


def centralize_gradients_(parameters, *, min_rank=2):
    """Center gradients transactionally while preserving parameter trainability."""

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
            return changed
        except BaseException:
            _restore_trainability_and_gradients(
                materialized, trainability, gradients
            )
            raise
