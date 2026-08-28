"""Read-only diagnostics for parameter changes across optimizer steps.

``ParameterUpdateTracker`` binds to a fixed Tensor collection, snapshots its
values, and later reports how much those same Tensor objects changed.  The
tracker does not participate in autograd and never mutates Tensor data,
gradients, mutation versions, or NumPy RNG state.
"""

from numbers import Integral

import numpy as np

from .tensor import Tensor


_FLOAT_MAX = np.finfo(np.float64).max


def _normalise_entries(parameters):
    """Materialize one Tensor collection and determine named/unnamed mode."""
    if isinstance(parameters, Tensor):
        raw_entries = (parameters,)
    elif (
        isinstance(parameters, tuple)
        and len(parameters) == 2
        and isinstance(parameters[0], str)
        and isinstance(parameters[1], Tensor)
    ):
        raw_entries = (parameters,)
    else:
        try:
            raw_entries = tuple(parameters)
        except TypeError as exc:
            raise TypeError(
                "parameter update tracker expects a Tensor or iterable"
            ) from exc

    if not raw_entries:
        return "unnamed", ()

    named_flags = []
    for entry in raw_entries:
        named_flags.append(
            isinstance(entry, (tuple, list)) and len(entry) == 2
        )
    if any(named_flags) and not all(named_flags):
        raise TypeError("parameter update tracker cannot mix named and unnamed entries")

    seen_tensors = set()
    if all(named_flags):
        seen_names = set()
        entries = []
        for index, entry in enumerate(raw_entries):
            name, parameter = entry
            if not isinstance(name, str):
                raise TypeError(f"parameter update name {index} must be a string")
            if not isinstance(parameter, Tensor):
                raise TypeError(f"parameter update entry {index} must contain a Tensor")
            if name in seen_names:
                raise ValueError(f"duplicate parameter update name: {name!r}")
            identity = id(parameter)
            if identity in seen_tensors:
                raise ValueError("parameter update tracker cannot bind duplicate Tensors")
            seen_names.add(name)
            seen_tensors.add(identity)
            entries.append((name, parameter))
        entries.sort(key=lambda item: item[0])
        return "named", tuple(entries)

    entries = []
    for index, parameter in enumerate(raw_entries):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter update entry {index} must be a Tensor")
        identity = id(parameter)
        if identity in seen_tensors:
            raise ValueError("parameter update tracker cannot bind duplicate Tensors")
        seen_tensors.add(identity)
        entries.append((index, parameter))
    return "unnamed", tuple(entries)


def _snapshot(parameter, locator):
    data = np.asarray(parameter.data)
    if not np.isfinite(data).all():
        raise ValueError(
            f"parameter update baseline {locator!r} must contain only finite values"
        )
    return {
        "data": np.array(data, dtype=np.float64, copy=True),
        "requires_grad": bool(parameter.requires_grad),
        "version": int(parameter._version),
    }


def _scaled_l2(arrays):
    """Return (representable_norm_or_None, overflow, largest, scaled_norm)."""
    largest = 0.0
    materialized = []
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        materialized.append(values)
        if values.size:
            candidate = float(np.max(np.abs(values)))
            if candidate > largest:
                largest = candidate

    if largest == 0.0:
        return 0.0, False, 0.0, 0.0

    scaled_sumsq = 0.0
    with np.errstate(under="ignore"):
        for values in materialized:
            if not values.size:
                continue
            scaled = values / largest
            scaled_sumsq += float(np.sum(scaled * scaled, dtype=np.float64))
    scaled_norm = float(np.sqrt(scaled_sumsq))
    if scaled_norm > 0.0 and largest > _FLOAT_MAX / scaled_norm:
        return None, True, largest, scaled_norm
    return float(largest * scaled_norm), False, largest, scaled_norm


def _absolute_updates(before, current):
    """Compute finite absolute deltas without overflowing subtraction.

    Returns ``(values, overflow)``.  If any true elementwise absolute delta is
    larger than binary64 can represent, ``values`` is ``None`` and ``overflow``
    is true.  Both inputs must be finite float64 arrays with identical shape.
    """
    before = np.asarray(before, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    scale = np.maximum(np.abs(before), np.abs(current))
    normalised = np.zeros_like(scale, dtype=np.float64)
    nonzero = scale != 0.0
    if np.any(nonzero):
        with np.errstate(under="ignore"):
            left = before[nonzero] / scale[nonzero]
            right = current[nonzero] / scale[nonzero]
        normalised[nonzero] = np.abs(left - right)
        with np.errstate(under="ignore"):
            overflow = normalised[nonzero] > (_FLOAT_MAX / scale[nonzero])
        if np.any(overflow):
            return None, True

    with np.errstate(under="ignore"):
        values = normalised * scale
    return values, False


def _ratio(update_norm, baseline_norm, *, unavailable=False):
    if unavailable or update_norm is None or baseline_norm is None:
        return None, baseline_norm == 0.0 if baseline_norm is not None else False
    if baseline_norm == 0.0:
        return None, True
    return float(update_norm / baseline_norm), False


def _json_shape(shape):
    return [int(value) for value in shape]


class ParameterUpdateTracker:
    """Track value changes on a fixed ordered collection of Tensor objects.

    Parameters can be a single Tensor, an iterable of Tensors, or an iterable
    of ``(name, Tensor)`` pairs such as ``model.named_parameters()``.  Named
    entries are sorted by name for deterministic reports; unnamed entries keep
    positional order.

    The initial baseline must be finite.  ``report()`` deliberately does *not*
    reject non-finite current values: an optimizer that produced NaN/Inf should
    be diagnosable.  Numerical update magnitudes become unavailable instead of
    understating the update by ignoring those elements.
    """

    def __init__(self, parameters):
        mode, entries = _normalise_entries(parameters)
        snapshots = tuple(_snapshot(parameter, locator) for locator, parameter in entries)
        self._mode = mode
        self._entries = entries
        self._snapshots = snapshots

    @property
    def mode(self):
        return self._mode

    @property
    def tensor_count(self):
        return len(self._entries)

    def refresh(self):
        """Transactionally replace the baseline with current finite values."""
        snapshots = tuple(
            _snapshot(parameter, locator)
            for locator, parameter in self._entries
        )
        self._snapshots = snapshots
        return self

    def baseline_values(self):
        """Return independent copies of the current baseline arrays."""
        return tuple(np.array(snapshot["data"], copy=True) for snapshot in self._snapshots)

    def report(self):
        """Return a deterministic strict-JSON-friendly update report."""
        entry_reports = []
        baseline_arrays = []
        current_finite_arrays = []
        update_arrays = []

        total_baseline_elements = 0
        total_current_elements = 0
        comparable_elements = 0
        changed_elements = 0
        nonfinite_elements = 0
        changed_tensors = 0
        rewritten_tensors = 0
        mutated_tensors = 0
        shape_changed_tensors = 0
        requires_grad_changed_tensors = 0
        nonfinite_tensors = 0
        any_update_overflow = False
        global_max_update = 0.0
        global_max_update_overflow = False

        for (locator, parameter), snapshot in zip(self._entries, self._snapshots):
            before = snapshot["data"]
            current = np.asarray(parameter.data)
            baseline_arrays.append(before)
            total_baseline_elements += int(before.size)
            total_current_elements += int(current.size)

            baseline_l2, baseline_l2_overflow, _, _ = _scaled_l2((before,))
            version_now = int(parameter._version)
            version_delta = version_now - snapshot["version"]
            mutation_observed = version_delta != 0
            if mutation_observed:
                mutated_tensors += 1

            requires_grad_now = bool(parameter.requires_grad)
            requires_grad_changed = requires_grad_now != snapshot["requires_grad"]
            if requires_grad_changed:
                requires_grad_changed_tensors += 1

            common = {
                "status": None,
                "baseline_shape": _json_shape(before.shape),
                "current_shape": _json_shape(current.shape),
                "baseline_requires_grad": snapshot["requires_grad"],
                "current_requires_grad": requires_grad_now,
                "requires_grad_changed": requires_grad_changed,
                "baseline_version": snapshot["version"],
                "current_version": version_now,
                "version_delta": version_delta,
                "mutation_observed": mutation_observed,
                "baseline_element_count": int(before.size),
                "current_element_count": int(current.size),
                "changed_element_count": None,
                "changed_fraction": None,
                "nonfinite_current_count": None,
                "baseline_l2": baseline_l2,
                "baseline_l2_overflow": baseline_l2_overflow,
                "current_l2": None,
                "current_l2_overflow": False,
                "update_l2": None,
                "update_l2_overflow": False,
                "max_abs_update": None,
                "max_abs_update_overflow": False,
                "update_to_baseline_ratio": None,
                "baseline_zero": baseline_l2 == 0.0,
            }
            if self._mode == "named":
                common["name"] = locator
            else:
                common["index"] = int(locator)

            if current.shape != before.shape:
                shape_changed_tensors += 1
                common["status"] = "shape_changed"
                if np.isfinite(current).all():
                    current_l2, current_l2_overflow, _, _ = _scaled_l2((current,))
                    common["current_l2"] = current_l2
                    common["current_l2_overflow"] = current_l2_overflow
                    current_finite_arrays.append(np.asarray(current, dtype=np.float64))
                else:
                    count = int(np.count_nonzero(~np.isfinite(current)))
                    common["nonfinite_current_count"] = count
                    nonfinite_elements += count
                    nonfinite_tensors += 1
                entry_reports.append(common)
                continue

            comparable_elements += int(before.size)
            finite_mask = np.isfinite(current)
            nonfinite_count = int(current.size - np.count_nonzero(finite_mask))
            common["nonfinite_current_count"] = nonfinite_count
            if nonfinite_count:
                nonfinite_elements += nonfinite_count
                nonfinite_tensors += 1
                common["status"] = "nonfinite"
                changed_count = int(np.count_nonzero(np.not_equal(current, before)))
                common["changed_element_count"] = changed_count
                common["changed_fraction"] = (
                    0.0 if current.size == 0 else float(changed_count / current.size)
                )
                if changed_count:
                    changed_tensors += 1
                entry_reports.append(common)
                continue

            current64 = np.asarray(current, dtype=np.float64)
            current_finite_arrays.append(current64)
            current_l2, current_l2_overflow, _, _ = _scaled_l2((current64,))
            common["current_l2"] = current_l2
            common["current_l2_overflow"] = current_l2_overflow

            changed_mask = np.not_equal(current64, before)
            changed_count = int(np.count_nonzero(changed_mask))
            changed_elements += changed_count
            common["changed_element_count"] = changed_count
            common["changed_fraction"] = (
                0.0 if current64.size == 0 else float(changed_count / current64.size)
            )

            absolute_update, update_overflow = _absolute_updates(before, current64)
            if update_overflow:
                any_update_overflow = True
                global_max_update_overflow = True
                common["update_l2_overflow"] = True
                common["max_abs_update_overflow"] = True
            else:
                update_arrays.append(absolute_update)
                update_l2, update_l2_overflow, _, _ = _scaled_l2((absolute_update,))
                common["update_l2"] = update_l2
                common["update_l2_overflow"] = update_l2_overflow
                if update_l2_overflow:
                    any_update_overflow = True
                max_update = (
                    0.0
                    if absolute_update.size == 0
                    else float(np.max(absolute_update))
                )
                common["max_abs_update"] = max_update
                if max_update > global_max_update:
                    global_max_update = max_update
                ratio, baseline_zero = _ratio(
                    update_l2,
                    baseline_l2,
                    unavailable=update_l2_overflow or baseline_l2_overflow,
                )
                common["update_to_baseline_ratio"] = ratio
                common["baseline_zero"] = baseline_zero

            if changed_count:
                changed_tensors += 1
                common["status"] = "updated"
            elif mutation_observed:
                rewritten_tensors += 1
                common["status"] = "rewritten"
            else:
                common["status"] = "unchanged"
            entry_reports.append(common)

        baseline_l2, baseline_l2_overflow, _, _ = _scaled_l2(baseline_arrays)
        all_current_finite = nonfinite_tensors == 0
        structurally_stable = shape_changed_tensors == 0

        if all_current_finite:
            # Shape changes do not prevent a current-value norm.
            all_current_arrays = [np.asarray(parameter.data, dtype=np.float64) for _, parameter in self._entries]
            current_l2, current_l2_overflow, _, _ = _scaled_l2(all_current_arrays)
        else:
            current_l2 = None
            current_l2_overflow = False

        update_metrics_available = all_current_finite and structurally_stable
        if update_metrics_available:
            if global_max_update_overflow:
                update_l2 = None
                update_l2_overflow = True
                max_abs_update = None
            else:
                update_l2, update_l2_overflow, _, _ = _scaled_l2(update_arrays)
                max_abs_update = global_max_update
                if update_l2_overflow:
                    any_update_overflow = True
            update_ratio, baseline_zero = _ratio(
                update_l2,
                baseline_l2,
                unavailable=update_l2_overflow or baseline_l2_overflow,
            )
        else:
            update_l2 = None
            update_l2_overflow = False
            max_abs_update = None
            update_ratio = None
            baseline_zero = baseline_l2 == 0.0

        unchanged_tensors = len(self._entries) - changed_tensors - shape_changed_tensors
        return {
            "mode": self._mode,
            "tensor_count": len(self._entries),
            "baseline_element_count": total_baseline_elements,
            "current_element_count": total_current_elements,
            "comparable_element_count": comparable_elements,
            "changed_element_count": changed_elements,
            "changed_tensor_count": changed_tensors,
            "unchanged_tensor_count": unchanged_tensors,
            "rewritten_tensor_count": rewritten_tensors,
            "mutated_tensor_count": mutated_tensors,
            "shape_changed_tensor_count": shape_changed_tensors,
            "requires_grad_changed_tensor_count": requires_grad_changed_tensors,
            "nonfinite_tensor_count": nonfinite_tensors,
            "nonfinite_element_count": nonfinite_elements,
            "all_finite": all_current_finite,
            "structurally_stable": structurally_stable,
            "update_metrics_available": update_metrics_available,
            "baseline_l2": baseline_l2,
            "baseline_l2_overflow": baseline_l2_overflow,
            "current_l2": current_l2,
            "current_l2_overflow": current_l2_overflow,
            "update_l2": update_l2,
            "update_l2_overflow": update_l2_overflow,
            "max_abs_update": max_abs_update,
            "max_abs_update_overflow": global_max_update_overflow,
            "update_to_baseline_ratio": update_ratio,
            "baseline_zero": baseline_zero,
            "changed": (
                changed_tensors > 0
                or shape_changed_tensors > 0
                or requires_grad_changed_tensors > 0
                or nonfinite_tensors > 0
            ),
            "entries": entry_reports,
        }
