"""Public gradient-centralization guard for complete parameter/gradient metadata."""

import weakref

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
from .tensor import _VersionedArray, _no_backward


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
        children = parameter._children
        if type(children) is not tuple:
            raise TypeError(f"parameter {index} graph metadata must be a plain tuple")
        if children != ():
            raise ValueError(f"parameter {index} must be a leaf Tensor")
        if getattr(parameter, "_backward_fn", None) is not _no_backward:
            raise TypeError(
                f"parameter {index} backward metadata must be the leaf no-op closure"
            )
        if getattr(parameter, "_detached_by_no_grad", None) is not False:
            raise TypeError(f"parameter {index} detached provenance must be false")


def _validate_leaf_provenance(parameters, provenance):
    for index, (parameter, entry) in enumerate(zip(parameters, provenance)):
        children, backward_fn, detached_by_no_grad = entry
        if (
            type(parameter._children) is not tuple
            or parameter._children != children
            or parameter._backward_fn is not backward_fn
            or parameter._detached_by_no_grad is not detached_by_no_grad
        ):
            raise RuntimeError(
                f"leaf provenance changed for parameter {index} during centralization"
            )


def _validate_entry_versions(parameters):
    for index, parameter in enumerate(parameters):
        version = parameter._version
        if type(version) is not int:
            raise TypeError(
                f"parameter {index} mutation version must be a non-negative integer"
            )
        if version < 0:
            raise ValueError(f"parameter {index} mutation version must be non-negative")


def _validate_transaction_versions(parameters, versions):
    for index, (parameter, entry_version) in enumerate(zip(parameters, versions)):
        version = parameter._version
        if type(version) is not int or version != entry_version:
            raise RuntimeError(
                f"mutation version changed for parameter {index} during centralization"
            )


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
        if np.asarray(data).dtype != np.dtype(np.float64):
            raise TypeError(f"parameter {index} data must have dtype float64")
        owner_ref = getattr(data, "_owner_ref", None)
        if type(owner_ref) is not weakref.ReferenceType:
            raise TypeError(
                f"parameter {index} data ownership metadata must be a weak reference"
            )
        if owner_ref() is not parameter:
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


def _tensor_storage_owner(array, parameter_index):
    if not isinstance(array, _VersionedArray):
        return None
    owner_ref = getattr(array, "_owner_ref", None)
    if type(owner_ref) is not weakref.ReferenceType:
        raise TypeError(
            f"gradient for parameter {parameter_index} ownership metadata must be "
            "a weak reference"
        )
    owner = owner_ref()
    if owner is None:
        raise ValueError(
            f"gradient for parameter {parameter_index} ownership metadata must "
            "reference a live Tensor"
        )
    return owner


def _has_tensor_storage_owner_in_base_chain(array, parameter_index):
    """Return whether an ndarray view ultimately aliases valid Tensor storage."""

    current = getattr(array, "base", None)
    seen = set()
    while isinstance(current, np.ndarray):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        if _tensor_storage_owner(current, parameter_index) is not None:
            return True
        current = getattr(current, "base", None)
    return False


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

        if _tensor_storage_owner(gradient, index) is not None:
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

        # An ndarray subclass can preserve the same external Tensor storage in
        # its ``base`` chain while no longer being a _VersionedArray itself.
        # Follow that chain so a subclass view cannot bypass the ownership guard.
        if _has_tensor_storage_owner_in_base_chain(gradient, index):
            raise ValueError(
                f"gradient for parameter {index} must not use foreign Tensor-managed storage"
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
    parameters,
    trainability,
    versions,
    grad_shapes,
    data_owners,
    leaf_provenance,
    gradients,
):
    rollback_error = None
    for (
        parameter,
        requires_grad,
        version,
        grad_shape,
        owner_ref,
        provenance,
        gradient_state,
    ) in zip(
        parameters,
        trainability,
        versions,
        grad_shapes,
        data_owners,
        leaf_provenance,
        gradients,
    ):
        gradient, values, dtype, strides, writeable = gradient_state
        children, backward_fn, detached_by_no_grad = provenance
        try:
            parameter.requires_grad = requires_grad
            if type(parameter._version) is not int:
                parameter._version = version
            parameter._grad_shape = grad_shape
            parameter.data._owner_ref = owner_ref
            parameter._children = children
            parameter._backward_fn = backward_fn
            parameter._detached_by_no_grad = detached_by_no_grad
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
        versions = tuple(parameter._version for parameter in materialized)
        grad_shapes = tuple(parameter._grad_shape for parameter in materialized)
        leaf_provenance = tuple(
            (
                parameter._children,
                parameter._backward_fn,
                parameter._detached_by_no_grad,
            )
            for parameter in materialized
        )
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
            _validate_transaction_versions(materialized, versions)
            _validate_leaf_provenance(materialized, leaf_provenance)
            _validate_parameter_data_owners(materialized, data_owners)
            _validate_nonwritten_gradients(materialized, gradients, min_rank)
            return changed
        except BaseException:
            _restore_parameter_metadata_and_gradients(
                materialized,
                trainability,
                versions,
                grad_shapes,
                data_owners,
                leaf_provenance,
                gradients,
            )
            raise
