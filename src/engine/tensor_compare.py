"""Read-only numerical comparison for live Tensor collections.

The checkpoint helpers can compare serialized state, but experiments and
regression tests often need to compare two *live* models before either side is
written to disk.  This module aligns named Tensor collections by name (or
unnamed collections by position) and returns a deterministic, JSON-friendly
report with structural and numerical differences.
"""

from collections.abc import Iterable
import math
from numbers import Real

import numpy as np

from .tensor import Tensor


def _validate_tolerance(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-negative finite real number")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _materialize(collection, label):
    """Return ``(mode, records)`` while consuming an iterable exactly once."""
    if isinstance(collection, Tensor):
        raw = (collection,)
    else:
        if not isinstance(collection, Iterable):
            raise TypeError(
                f"{label} must be a Tensor or iterable of Tensors"
            )
        raw = tuple(collection)

    mode = None
    records = []
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
                raise TypeError(f"{label} Tensor names must be strings")
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"{label} named entries must contain Tensor values"
                )
        else:
            raise TypeError(
                f"{label} entries must be Tensors or (name, Tensor) pairs"
            )

        if mode is None:
            mode = item_mode
        elif item_mode != mode:
            raise TypeError(f"{label} cannot mix named and unnamed entries")

        tensor_id = id(tensor)
        if tensor_id in seen_tensors:
            raise ValueError(f"{label} must not contain duplicate Tensors")
        seen_tensors.add(tensor_id)

        if name is not None:
            if name in seen_names:
                raise ValueError(f"{label} Tensor names must be unique")
            seen_names.add(name)

        records.append((index, name, tensor))
    return mode, records


def _json_finite_float(value):
    """Return ``(binary64_value, overflow)`` without leaking Infinity."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(value)
    except OverflowError:
        return None, True
    if not math.isfinite(value):
        return None, True
    return value, False


def _numerical_metrics(left, right, *, atol, rtol, equal_nan):
    """Compare same-shape, same-dtype arrays without emitting non-finite JSON."""
    size = int(left.size)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        close = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=equal_nan)
    close_count = int(np.count_nonzero(close))
    mismatch_count = size - close_count

    finite_pair = np.isfinite(left) & np.isfinite(right)
    finite_pair_count = int(np.count_nonzero(finite_pair))
    finite_mismatch_count = int(np.count_nonzero(finite_pair & ~close))
    nonfinite_mismatch_count = mismatch_count - finite_mismatch_count

    max_abs_diff = None
    max_abs_diff_overflow = False
    max_symmetric_relative_diff = None
    if finite_pair_count:
        left_finite = left[finite_pair]
        right_finite = right[finite_pair]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            absolute = np.abs(left_finite - right_finite)
        if not np.all(np.isfinite(absolute)):
            max_abs_diff_overflow = True
        else:
            max_abs_diff, max_abs_diff_overflow = _json_finite_float(
                np.max(absolute)
            )

        # A symmetric scale avoids overflow from |a-b| while remaining useful
        # for diagnostics.  For nonzero finite pairs this is
        # |a/scale - b/scale| with scale=max(|a|, |b|), so the value is in [0, 2].
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scale = np.maximum(np.abs(left_finite), np.abs(right_finite))
            relative = np.zeros(scale.shape, dtype=np.float64)
            nonzero = scale != 0
            if np.any(nonzero):
                relative[nonzero] = np.abs(
                    left_finite[nonzero] / scale[nonzero]
                    - right_finite[nonzero] / scale[nonzero]
                )
        max_symmetric_relative_diff = float(np.max(relative))

    return {
        "compared_elements": size,
        "close_elements": close_count,
        "mismatch_elements": mismatch_count,
        "finite_pair_elements": finite_pair_count,
        "finite_mismatch_elements": finite_mismatch_count,
        "nonfinite_mismatch_elements": nonfinite_mismatch_count,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_overflow": max_abs_diff_overflow,
        "max_symmetric_relative_diff": max_symmetric_relative_diff,
    }


def _missing_entry(*, index=None, name=None, side):
    entry = {
        "status": "mismatch",
        "issues": [f"missing_{side}"],
        "allclose": False,
        "compared_elements": 0,
        "close_elements": 0,
        "mismatch_elements": 0,
        "finite_pair_elements": 0,
        "finite_mismatch_elements": 0,
        "nonfinite_mismatch_elements": 0,
        "max_abs_diff": None,
        "max_abs_diff_overflow": False,
        "max_symmetric_relative_diff": None,
    }
    if name is not None:
        entry["name"] = name
    else:
        entry["index"] = index
    return entry


def _compare_pair(left, right, *, index=None, name=None, atol, rtol, equal_nan):
    left_data = np.asarray(left.data)
    right_data = np.asarray(right.data)
    issues = []
    if bool(left.requires_grad) != bool(right.requires_grad):
        issues.append("requires_grad")
    if left_data.shape != right_data.shape:
        issues.append("shape")
    if left_data.dtype != right_data.dtype:
        issues.append("dtype")

    entry = {
        "left_shape": list(left_data.shape),
        "right_shape": list(right_data.shape),
        "left_dtype": str(left_data.dtype),
        "right_dtype": str(right_data.dtype),
        "left_requires_grad": bool(left.requires_grad),
        "right_requires_grad": bool(right.requires_grad),
        "compared_elements": 0,
        "close_elements": 0,
        "mismatch_elements": 0,
        "finite_pair_elements": 0,
        "finite_mismatch_elements": 0,
        "nonfinite_mismatch_elements": 0,
        "max_abs_diff": None,
        "max_abs_diff_overflow": False,
        "max_symmetric_relative_diff": None,
    }
    if name is not None:
        entry["name"] = name
    else:
        entry["index"] = index

    if left_data.shape == right_data.shape and left_data.dtype == right_data.dtype:
        metrics = _numerical_metrics(
            left_data,
            right_data,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )
        entry.update(metrics)
        if metrics["mismatch_elements"]:
            issues.append("values")

    entry["issues"] = issues
    entry["allclose"] = not issues
    entry["status"] = "match" if not issues else "mismatch"
    return entry


def compare_tensor_collections(
    first,
    second,
    *,
    atol=0.0,
    rtol=0.0,
    equal_nan=False,
):
    """Compare two live Tensor collections and return a JSON-friendly report.

    Each side may be a single :class:`Tensor`, an iterable of Tensors, or an
    iterable of ``(name, Tensor)`` pairs such as ``model.named_parameters()``.
    Non-empty collections must use the same naming mode. Named collections are
    aligned by name and are therefore insensitive to traversal order; unnamed
    collections are aligned by position.

    Structural identity includes shape, dtype, and ``requires_grad``. Values are
    compared with NumPy ``isclose`` semantics using non-negative finite ``atol``
    and ``rtol``. ``equal_nan`` is opt-in.  Numerical metrics are only computed
    when shape and dtype match.

    The function is observational only: it never runs backward, mutates Tensor
    data/gradients, or touches NumPy's global RNG state.
    """
    atol = _validate_tolerance("atol", atol)
    rtol = _validate_tolerance("rtol", rtol)
    if not isinstance(equal_nan, (bool, np.bool_)):
        raise TypeError("equal_nan must be a boolean")
    equal_nan = bool(equal_nan)

    first_mode, first_records = _materialize(first, "first tensor collection")
    second_mode, second_records = _materialize(second, "second tensor collection")
    if first_mode is not None and second_mode is not None and first_mode != second_mode:
        raise ValueError("tensor collection naming modes must match")
    mode = first_mode or second_mode or "unnamed"

    entries = []
    if mode == "named":
        first_by_name = {name: tensor for _, name, tensor in first_records}
        second_by_name = {name: tensor for _, name, tensor in second_records}
        for name in sorted(set(first_by_name) | set(second_by_name)):
            if name not in first_by_name:
                entries.append(_missing_entry(name=name, side="left"))
            elif name not in second_by_name:
                entries.append(_missing_entry(name=name, side="right"))
            else:
                entries.append(
                    _compare_pair(
                        first_by_name[name],
                        second_by_name[name],
                        name=name,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=equal_nan,
                    )
                )
    else:
        first_tensors = [tensor for _, _, tensor in first_records]
        second_tensors = [tensor for _, _, tensor in second_records]
        for index in range(max(len(first_tensors), len(second_tensors))):
            if index >= len(first_tensors):
                entries.append(_missing_entry(index=index, side="left"))
            elif index >= len(second_tensors):
                entries.append(_missing_entry(index=index, side="right"))
            else:
                entries.append(
                    _compare_pair(
                        first_tensors[index],
                        second_tensors[index],
                        index=index,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=equal_nan,
                    )
                )

    representable_abs = [
        entry["max_abs_diff"]
        for entry in entries
        if entry["max_abs_diff"] is not None
    ]
    aggregate_abs_overflow = any(entry["max_abs_diff_overflow"] for entry in entries)
    relative_values = [
        entry["max_symmetric_relative_diff"]
        for entry in entries
        if entry["max_symmetric_relative_diff"] is not None
    ]

    return {
        "mode": mode,
        "atol": atol,
        "rtol": rtol,
        "equal_nan": equal_nan,
        "left_tensor_count": len(first_records),
        "right_tensor_count": len(second_records),
        "entry_count": len(entries),
        "matching_tensor_count": sum(entry["allclose"] for entry in entries),
        "mismatching_tensor_count": sum(not entry["allclose"] for entry in entries),
        "missing_left_count": sum("missing_left" in entry["issues"] for entry in entries),
        "missing_right_count": sum("missing_right" in entry["issues"] for entry in entries),
        "shape_mismatch_count": sum("shape" in entry["issues"] for entry in entries),
        "dtype_mismatch_count": sum("dtype" in entry["issues"] for entry in entries),
        "requires_grad_mismatch_count": sum(
            "requires_grad" in entry["issues"] for entry in entries
        ),
        "value_mismatch_tensor_count": sum("values" in entry["issues"] for entry in entries),
        "compared_element_count": sum(entry["compared_elements"] for entry in entries),
        "mismatch_element_count": sum(entry["mismatch_elements"] for entry in entries),
        "finite_mismatch_element_count": sum(
            entry["finite_mismatch_elements"] for entry in entries
        ),
        "nonfinite_mismatch_element_count": sum(
            entry["nonfinite_mismatch_elements"] for entry in entries
        ),
        "max_abs_diff": None
        if aggregate_abs_overflow
        else max(representable_abs, default=None),
        "max_abs_diff_overflow": aggregate_abs_overflow,
        "max_symmetric_relative_diff": max(relative_values, default=None),
        "allclose": all(entry["allclose"] for entry in entries),
        "entries": entries,
    }
