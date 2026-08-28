"""Flatten Tensor parameters to vectors and restore them safely."""

import numpy as np

from .tensor import Tensor


def _materialize_parameters(parameters):
    try:
        parameters = tuple(parameters)
    except TypeError as exc:
        raise TypeError("parameters must be an iterable of Tensors") from exc

    seen = set()
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor references")
        seen.add(marker)
    return parameters


def _prepare_vector(vector):
    try:
        raw = np.asarray(vector)
    except (TypeError, ValueError) as exc:
        raise TypeError("vector must contain real numeric values") from exc

    if raw.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got shape {raw.shape}")

    is_integer = np.issubdtype(raw.dtype, np.integer)
    is_floating = np.issubdtype(raw.dtype, np.floating)
    if np.issubdtype(raw.dtype, np.bool_) or not (is_integer or is_floating):
        raise TypeError("vector must contain real numeric values")
    if not np.isfinite(raw).all():
        raise ValueError("vector must contain only finite values")

    # Tensor storage is float64. Preflight the complete conversion before any
    # destination write so a finite wider-dtype value cannot partially restore
    # parameters and only then overflow while being narrowed to float64.
    with np.errstate(over="ignore", invalid="ignore"):
        prepared = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(prepared).all():
        raise ValueError("vector values must be representable as finite Tensor data")
    return prepared


def parameters_to_vector(parameters):
    """Return an independent flat float64 snapshot of Tensor parameter data.

    Parameters are concatenated in iterable order. Scalars contribute one
    element and empty tensors contribute zero elements. Duplicate Tensor
    references are rejected because they do not define an unambiguous inverse
    mapping for :func:`vector_to_parameters_`.
    """
    parameters = _materialize_parameters(parameters)
    if not parameters:
        return np.empty(0, dtype=np.float64)

    pieces = [np.asarray(parameter.data).reshape(-1) for parameter in parameters]
    return np.concatenate(pieces).astype(np.float64, copy=False)


def vector_to_parameters_(vector, parameters):
    """Copy one flat vector into Tensor parameters in iterable order.

    The vector and complete destination collection are validated before any
    Tensor storage is changed. Successful non-empty writes go through the
    public tracked ``Tensor.data`` array, so existing forward graphs that used
    those parameters are invalidated by the normal mutation-version mechanism.
    """
    prepared = _prepare_vector(vector)
    parameters = _materialize_parameters(parameters)

    expected = sum(parameter.data.size for parameter in parameters)
    if prepared.size != expected:
        raise ValueError(
            f"vector length mismatch: expected {expected}, got {prepared.size}"
        )

    for index, parameter in enumerate(parameters):
        if not parameter.data.flags.writeable:
            raise ValueError(f"parameter {index} data must be writeable")

    offset = 0
    for parameter in parameters:
        size = parameter.data.size
        if size:
            values = prepared[offset : offset + size].reshape(parameter.shape)
            parameter.data[...] = values
        offset += size
