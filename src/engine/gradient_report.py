"""Read-only gradient health diagnostics for Tensor collections.

The training loop needs a fast scalar global norm, while debugging often needs
more context: which parameter is missing a gradient, where NaN/Inf appeared,
or whether a whole gradient buffer collapsed to zero. This module provides a
standalone, side-effect-free report without changing optimizer or Tensor state.
"""

from collections.abc import Iterable
import math

import numpy as np

from .tensor import Tensor


_ANOMALOUS_STATUSES = {"missing", "invalid_type", "shape_mismatch", "nonfinite"}


def _materialize_parameters(parameters):
    """Normalize Tensor or iterable input to ``(name, tensor)`` records."""
    if isinstance(parameters, Tensor):
        raw = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError(
                "gradient report parameters must be a Tensor or iterable of Tensors"
            )
        raw = tuple(parameters)

    records = []
    mode = None
    seen_tensors = set()
    seen_names = set()
    for index, item in enumerate(raw):
        if isinstance(item, Tensor):
            item_mode = "unnamed"
            name = None
            tensor = item
        elif isinstance(item, tuple) and len(item) == 2:
            item_mode = "named"
            name, tensor = item
            if not isinstance(name, str):
                raise TypeError("gradient report parameter names must be strings")
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    "gradient report named entries must contain Tensor values"
                )
        else:
            raise TypeError(
                "gradient report entries must be Tensors or (name, Tensor) pairs"
            )

        if mode is None:
            mode = item_mode
        elif item_mode != mode:
            raise TypeError("gradient report cannot mix named and unnamed entries")

        tensor_id = id(tensor)
        if tensor_id in seen_tensors:
            raise ValueError("gradient report parameters must not contain duplicates")
        seen_tensors.add(tensor_id)

        if name is not None:
            if name in seen_names:
                raise ValueError("gradient report parameter names must be unique")
            seen_names.add(name)

        records.append((index, name, tensor))
    return records


def _finite_float(value):
    """Convert a finite NumPy scalar to binary64, reporting range overflow."""
    try:
        result = float(value)
    except OverflowError:
        return None, True
    if not math.isfinite(result):
        return None, True
    return result, False


def _stable_l2(arrays):
    """Return a warning-free binary64 L2 norm and an overflow flag."""
    arrays = tuple(arrays)
    if not arrays:
        return 0.0, False

    scales = []
    for array in arrays:
        if array.size:
            with np.errstate(over="ignore", invalid="ignore"):
                scales.append(np.max(np.abs(array)))
    if not scales:
        return 0.0, False

    scale = max(scales)
    if scale == 0:
        return 0.0, False
    scale_float, overflow = _finite_float(scale)
    if overflow:
        return None, True

    sum_squares = 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for array in arrays:
            if not array.size:
                continue
            scaled = np.asarray(array, dtype=np.float64) / scale_float
            sum_squares += float(np.sum(scaled * scaled, dtype=np.float64))
        norm = scale_float * math.sqrt(sum_squares)
    if not math.isfinite(norm):
        return None, True
    return norm, False


def _gradient_entry(index, name, tensor):
    grad = tensor.grad
    base = {
        "index": index,
        "name": name,
        "shape": list(tensor.shape),
        "requires_grad": bool(tensor.requires_grad),
        "parameter_elements": int(tensor.data.size),
        "gradient_present": grad is not None,
        "gradient_type": None if grad is None else type(grad).__name__,
        "gradient_dtype": None,
        "gradient_shape": None,
        "gradient_elements": 0,
        "finite_elements": 0,
        "zero_elements": 0,
        "nan_elements": 0,
        "posinf_elements": 0,
        "neginf_elements": 0,
        "max_finite_abs": None,
        "magnitude_overflow": False,
        "l2_norm": None,
        "l2_overflow": False,
        "unexpected_for_frozen": bool(not tensor.requires_grad and grad is not None),
    }

    if grad is None:
        base["status"] = "missing" if tensor.requires_grad else "not_required"
        base["anomaly"] = tensor.requires_grad
        return base

    if not isinstance(grad, np.ndarray) or not np.issubdtype(grad.dtype, np.floating):
        base["status"] = "invalid_type"
        base["anomaly"] = True
        return base

    base["gradient_dtype"] = str(grad.dtype)
    base["gradient_shape"] = list(grad.shape)
    if grad.shape != tensor.shape:
        base["gradient_elements"] = int(grad.size)
        base["status"] = "shape_mismatch"
        base["anomaly"] = True
        return base

    size = int(grad.size)
    base["gradient_elements"] = size
    nan_count = int(np.count_nonzero(np.isnan(grad)))
    posinf_count = int(np.count_nonzero(np.isposinf(grad)))
    neginf_count = int(np.count_nonzero(np.isneginf(grad)))
    finite_mask = np.isfinite(grad)
    finite_count = int(np.count_nonzero(finite_mask))
    zero_count = int(np.count_nonzero(finite_mask & (grad == 0)))
    base.update(
        finite_elements=finite_count,
        zero_elements=zero_count,
        nan_elements=nan_count,
        posinf_elements=posinf_count,
        neginf_elements=neginf_count,
    )

    if finite_count:
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = np.max(np.abs(grad[finite_mask]))
        base["max_finite_abs"], base["magnitude_overflow"] = _finite_float(max_abs)

    nonfinite_count = nan_count + posinf_count + neginf_count
    if nonfinite_count:
        base["status"] = "nonfinite"
        base["anomaly"] = True
        return base

    base["l2_norm"], base["l2_overflow"] = _stable_l2((grad,))
    if size > 0 and zero_count == size:
        base["status"] = "zero"
    else:
        base["status"] = "finite"
    base["anomaly"] = base["unexpected_for_frozen"] or base["l2_overflow"]
    return base


def gradient_report(parameters):
    """Return deterministic, JSON-friendly diagnostics for Tensor gradients.

    ``parameters`` may be a Tensor, an iterable of Tensors, or an iterable of
    ``(name, Tensor)`` pairs such as ``model.named_parameters()``. Named and
    unnamed entries cannot be mixed, Tensor identities cannot be repeated, and
    names must be unique.

    The report is observational only: no gradient buffer is created, replaced,
    or mutated; no backward closure runs; and NumPy RNG state is untouched.
    Missing gradients on trainable tensors, malformed buffers, non-finite
    values, L2 overflow, and gradients attached to frozen tensors are marked as
    anomalies. A zero but finite trainable gradient is reported separately and
    is not itself considered anomalous.
    """
    records = _materialize_parameters(parameters)
    entries = [_gradient_entry(index, name, tensor) for index, name, tensor in records]

    trainable_arrays = []
    trainable_norm_invalid = False
    trainable_norm_overflow = False
    for entry, (_, _, tensor) in zip(entries, records):
        if not tensor.requires_grad:
            continue
        if entry["status"] in _ANOMALOUS_STATUSES:
            trainable_norm_invalid = True
            continue
        if entry["l2_overflow"]:
            trainable_norm_overflow = True
            continue
        trainable_arrays.append(tensor.grad)

    if trainable_norm_invalid:
        trainable_global_l2_norm = None
        trainable_global_l2_overflow = False
    elif trainable_norm_overflow:
        trainable_global_l2_norm = None
        trainable_global_l2_overflow = True
    else:
        trainable_global_l2_norm, trainable_global_l2_overflow = _stable_l2(
            trainable_arrays
        )

    finite_abs_values = [
        entry["max_finite_abs"]
        for entry in entries
        if entry["max_finite_abs"] is not None
    ]
    max_finite_abs_gradient_overflow = any(
        entry["magnitude_overflow"] for entry in entries
    )
    if max_finite_abs_gradient_overflow:
        max_finite_abs_gradient = None
    else:
        max_finite_abs_gradient = max(finite_abs_values, default=None)

    return {
        "parameter_count": len(entries),
        "trainable_parameter_count": sum(entry["requires_grad"] for entry in entries),
        "frozen_parameter_count": sum(not entry["requires_grad"] for entry in entries),
        "parameter_element_count": sum(entry["parameter_elements"] for entry in entries),
        "gradient_present_count": sum(entry["gradient_present"] for entry in entries),
        "missing_gradient_count": sum(entry["status"] == "missing" for entry in entries),
        "unexpected_gradient_count": sum(
            entry["unexpected_for_frozen"] for entry in entries
        ),
        "invalid_gradient_count": sum(
            entry["status"] in {"invalid_type", "shape_mismatch"} for entry in entries
        ),
        "nonfinite_gradient_count": sum(
            entry["status"] == "nonfinite" for entry in entries
        ),
        "zero_gradient_count": sum(entry["status"] == "zero" for entry in entries),
        "anomaly_count": sum(entry["anomaly"] for entry in entries),
        "gradient_element_count": sum(
            entry["gradient_elements"]
            for entry in entries
            if entry["status"] not in {"invalid_type", "shape_mismatch"}
        ),
        "finite_element_count": sum(entry["finite_elements"] for entry in entries),
        "zero_element_count": sum(entry["zero_elements"] for entry in entries),
        "nan_element_count": sum(entry["nan_elements"] for entry in entries),
        "posinf_element_count": sum(entry["posinf_elements"] for entry in entries),
        "neginf_element_count": sum(entry["neginf_elements"] for entry in entries),
        "max_finite_abs_gradient": max_finite_abs_gradient,
        "max_finite_abs_gradient_overflow": max_finite_abs_gradient_overflow,
        "trainable_global_l2_norm": trainable_global_l2_norm,
        "trainable_global_l2_overflow": trainable_global_l2_overflow,
        "entries": entries,
    }