"""Public gradient-centralization guard for complete parameter/gradient metadata."""

import numpy as np

from .gradient_centralization import (
    _CENTRALIZATION_LOCK,
    _centralized_candidate,
    _materialize_parameters,
    _restore_array_dtype,
    _restore_array_shape,
    _restore_array_strides,
    _restore_array_writeable,
    _validate_min_rank,
    centralize_gradients_ as _centralize_gradients_impl,
)
from .tensor import _VersionedArray


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


def _validate_leaf_parameters(parameters):
    for index, parameter in enumerate(parameters):
        if parameter._children:
            raise ValueError(f"parameter {index} must be a leaf Tensor")


def _validate_entry_versions(parameters):
    for index, parameter in enumerate(parameters):
        version = parameter._version
        if type(version) is not int:
            raise TypeError(
                f"parameter {index} mutation version must be a non-negative integer"
            )
        if version < 0:
            raise ValueError(f"parameter {index} mutation version must be non-negative")


def _validate_entry_grad_shapes(parameters):
    for index, parameter in enumerate(parameters):
        expected = np.asarray(parameter.data).shape
        if type(parameter._grad_shape) is not tuple or parameter._grad_shape != expected:
            raise ValueError(
                f"parameter {index} gradient shape metadata must match parameter data shape"
            )


def _snapshot_parameter_data_owners(parameters):
    owners = []
    for index, parameter in enumerate(parameters):
        data = parameter.data
        if not isinstance(data, _VersionedArray):
            raise TypeError(f"parameter {index} data must use Tensor-managed storage")
        owner_ref = getattr(data, "_owner_ref", None)
        if owner_ref is None or owner_ref() is not parameter:
            raise ValueError(
                f"parameter {index} data ownership metadata must reference its Tensor"
            )
        owners.append(owner_ref)
    return tuple(owners)


def _validate_parameter_data_owners(parameters, owners):
    for index, (parameter, owner_ref) in enumerate(zip(parameters, owners)):
        current = getattr(parameter.data, "_owner_ref", None)
        if current is not owner_ref or current() is not parameter:
            raise RuntimeError(
                f"parameter data ownership changed for parameter {index} "
                "during centralization"
            )


def _has_live_tensor_storage_owner(array):
    if not isinstance(array, _VersionedArray):
        return False
    owner_ref = getattr(array, "_owner_ref", None)
    return owner_ref is not None and owner_ref() is not None


def _validate_foreign_tensor_managed_gradients(parameters, min_rank):
    """Reject writes whose storage ownership cannot be proven local and safe."""

    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        requires_grad = parameter.requires_grad
        if not isinstance(requires_grad, bool) or not requires_grad:
            continue
        if not isinstance(gradient, np.ndarray):
            continue

        base = np.asarray(gradient)
        parameter_data = np.asarray(parameter.data)
        if base.shape != parameter_data.shape:
            continue
        if not np.issubdtype(base.dtype, np.floating) or not np.all(np.isfinite(base)):
            continue
        if parameter_data.ndim < min_rank:
            continue

        candidate = _centralized_candidate(base)
        if np.array_equal(base, candidate):
            continue

        overlaps_bound_data = False
        for bound_parameter in parameters:
            try:
                if np.shares_memory(base, np.asarray(bound_parameter.data)):
                    overlaps_bound_data = True
                    break
            except ValueError:
                overlaps_bound_data = True
                break
        if overlaps_bound_data:
            continue

        if _has_live_tensor_storage_owner(gradient):
            raise ValueError(
                f"gradient for parameter {index} must not use foreign Tensor-managed storage"
            )

        # ``external.data.view(np.ndarray)`` deliberately strips the
        # _VersionedArray subclass (and therefore its weak owner metadata) while
        # retaining the same writable storage. Exact ordinary ndarray views are
        # consequently ownership-ambiguous: the helper cannot prove that an
        # in-place write is confined to caller-owned gradient storage. Fail
        # closed only when centralization would actually write; independent
        # owning ndarrays and exact no-ops remain accepted.
        if type(gradient) is np.ndarray and gradient.base is not None:
            raise ValueError(
                f"gradient for parameter {index} must own its storage when "
                "centralization requires a write"
            )


def _validate_nonwritten_gradients(parameters, gradients, min_rank):
    for index, (parameter, gradient_state) in enumerate(zip(parameters, gradients)):
        gradient, values, dtype, strides, writeable = gradient_state
        if values is not None and np.asarray(parameter.data).ndim >= min_rank:
            continue
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


def _restore_parameter_metadata_and_gradients(
    parameters, trainability, grad_shapes, data_owners, gradients
):
    rollback_error = None
    for parameter, requires_grad, grad_shape, owner_ref, gradient_state in zip(
        parameters, trainability, grad_shapes, data_owners, gradients
    ):
        gradient, values, dtype, strides, writeable = gradient_state
        try:
            parameter.requires_grad = requires_grad
            parameter._grad_shape = grad_shape
            parameter.data._owner_ref = owner_ref
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
    """Center gradients transactionally while preserving non-written state."""

    min_rank = _validate_min_rank(min_rank)
    with _CENTRALIZATION_LOCK:
        materialized = _materialize_parameters(parameters)
        _validate_leaf_parameters(materialized)
        _validate_entry_versions(materialized)
        _validate_entry_grad_shapes(materialized)
        data_owners = _snapshot_parameter_data_owners(materialized)
        _validate_foreign_tensor_managed_gradients(materialized, min_rank)
        trainability = tuple(parameter.requires_grad for parameter in materialized)
        grad_shapes = tuple(parameter._grad_shape for parameter in materialized)
        gradients = _snapshot_gradients(materialized)
        try:
            changed = _centralize_gradients_impl(materialized, min_rank=min_rank)
            for index, (parameter, requires_grad, grad_shape) in enumerate(
                zip(materialized, trainability, grad_shapes)
            ):
                if parameter.requires_grad is not requires_grad:
                    raise RuntimeError(
                        f"parameter trainability changed for parameter {index} "
                        "during centralization"
                    )
                if parameter._grad_shape != grad_shape:
                    raise RuntimeError(
                        f"gradient shape metadata changed for parameter {index} "
                        "during centralization"
                    )
            _validate_parameter_data_owners(materialized, data_owners)
            _validate_nonwritten_gradients(materialized, gradients, min_rank)
            return changed
        except BaseException:
            _restore_parameter_metadata_and_gradients(
                materialized, trainability, grad_shapes, data_owners, gradients
            )
            raise
