"""Deterministic, JSON-friendly health reports for live Tensor values.

The report is observational only: Tensor data, gradient buffers, mutation versions,
and NumPy RNG state are never modified. Large finite values use scale-normalized L2
accumulation so diagnostics stay warning-neutral under ``-W error``.
"""

from collections.abc import Iterable
import math

import numpy as np

from .tensor import Tensor


def _materialize(parameters):
    if isinstance(parameters, Tensor):
        raw = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("parameters must be a Tensor or iterable of Tensors")
        raw = tuple(parameters)

    named = None
    entries = []
    seen_tensors = set()
    seen_names = set()
    for index, item in enumerate(raw):
        if isinstance(item, Tensor):
            item_named = False
            name = str(index)
            tensor = item
        elif isinstance(item, tuple) and len(item) == 2:
            item_named = True
            name, tensor = item
            if not isinstance(name, str):
                raise TypeError(f"parameter name {index} must be a string")
        else:
            raise TypeError(
                "parameters must contain Tensors or (name, Tensor) pairs"
            )

        if not isinstance(tensor, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        if named is None:
            named = item_named
        elif named != item_named:
            raise ValueError("parameters must not mix named and unnamed entries")

        marker = id(tensor)
        if marker in seen_tensors:
            raise ValueError("parameters must not contain duplicate Tensors")
        seen_tensors.add(marker)
        if item_named:
            if name in seen_names:
                raise ValueError(f"duplicate parameter name {name!r}")
            seen_names.add(name)
        entries.append((name, tensor))

    if named:
        entries.sort(key=lambda pair: pair[0])
    return tuple(entries), bool(named)


def _stable_l2(arrays):
    scale = 0.0
    for array in arrays:
        if array.size == 0:
            continue
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size == 0:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            local = float(np.max(np.abs(finite)))
        if local > scale:
            scale = local
    if scale == 0.0:
        return 0.0, False

    parts = []
    with np.errstate(over="raise", invalid="raise", under="ignore", divide="raise"):
        for array in arrays:
            finite = np.asarray(array)[np.isfinite(array)]
            if finite.size == 0:
                continue
            scaled = finite / scale
            parts.append(float(np.sum(scaled * scaled, dtype=np.float64)))
    squared = math.fsum(parts)
    root = math.sqrt(squared)
    if root == 0.0:
        return 0.0, False
    limit = np.finfo(np.float64).max
    if scale > limit / root:
        return None, True
    return float(scale * root), False


def _entry(name, tensor):
    data = np.asarray(tensor.data)
    finite_mask = np.isfinite(data)
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = int(data.size - finite_count)
    nan_count = int(np.count_nonzero(np.isnan(data)))
    posinf_count = int(np.count_nonzero(np.isposinf(data)))
    neginf_count = int(np.count_nonzero(np.isneginf(data)))
    zero_count = int(np.count_nonzero(data == 0.0))

    if finite_count:
        finite = data[finite_mask]
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = float(np.max(np.abs(finite)))
    else:
        minimum = None
        maximum = None
        max_abs = None

    if nonfinite_count:
        l2 = None
        l2_overflow = False
    else:
        l2, l2_overflow = _stable_l2((data,))

    return {
        "name": name,
        "shape": list(tensor.shape),
        "element_count": int(data.size),
        "requires_grad": bool(tensor.requires_grad),
        "mutation_version": int(tensor._version),
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "zero_count": zero_count,
        "min_finite": minimum,
        "max_finite": maximum,
        "max_abs_finite": max_abs,
        "l2": l2,
        "l2_overflow": l2_overflow,
    }


def parameter_report(parameters):
    """Return a deterministic health report for live Tensor values.

    A single Tensor, an iterable of Tensors, or an iterable of ``(name, Tensor)``
    pairs is accepted. Named inputs are sorted by name; unnamed inputs preserve their
    positional order. Duplicate Tensor identities and duplicate names are rejected.

    Non-finite values are diagnosed rather than raised. Their exact NaN/+Inf/-Inf
    counts remain visible, while aggregate L2 becomes unavailable instead of emitting
    JSON ``NaN`` or ``Infinity``. A mathematically finite-input L2 that exceeds the
    binary64 range is represented as ``None`` with ``l2_overflow=True``.
    """
    entries, named = _materialize(parameters)
    tensors = tuple(tensor for _, tensor in entries)
    details = tuple(_entry(name, tensor) for name, tensor in entries)

    element_count = sum(item["element_count"] for item in details)
    finite_count = sum(item["finite_count"] for item in details)
    nonfinite_count = sum(item["nonfinite_count"] for item in details)
    nan_count = sum(item["nan_count"] for item in details)
    posinf_count = sum(item["positive_infinity_count"] for item in details)
    neginf_count = sum(item["negative_infinity_count"] for item in details)
    zero_count = sum(item["zero_count"] for item in details)

    finite_magnitudes = [
        item["max_abs_finite"]
        for item in details
        if item["max_abs_finite"] is not None
    ]
    max_abs = max(finite_magnitudes) if finite_magnitudes else None
    if nonfinite_count:
        l2 = None
        l2_overflow = False
    else:
        l2, l2_overflow = _stable_l2(tuple(tensor.data for tensor in tensors))

    return {
        "named": named,
        "tensor_count": len(details),
        "element_count": element_count,
        "trainable_tensor_count": sum(
            1 for _, tensor in entries if tensor.requires_grad
        ),
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nan_count": nan_count,
        "positive_infinity_count": posinf_count,
        "negative_infinity_count": neginf_count,
        "zero_count": zero_count,
        "max_abs_finite": max_abs,
        "l2": l2,
        "l2_overflow": l2_overflow,
        "parameters": list(details),
    }
