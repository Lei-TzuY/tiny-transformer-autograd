"""Read-only multi-task gradient conflict diagnostics."""

from collections.abc import Iterable
import math
import threading

import numpy as np

from .tensor import Tensor


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        values = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("parameters must be a Tensor or iterable of Tensors")
        values = tuple(parameters)

    seen = set()
    for parameter in values:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameters must contain only Tensors")
        if id(parameter) in seen:
            raise ValueError("parameters must not contain duplicate Tensors")
        seen.add(id(parameter))
        if not parameter.requires_grad:
            raise ValueError("all parameters must require gradients")
    return values


def _gradient_snapshot(parameter, index):
    gradient = parameter.grad
    if gradient is None:
        return np.zeros(parameter.shape, dtype=np.float64)
    if not isinstance(gradient, np.ndarray):
        raise TypeError(f"gradient {index} must be a NumPy array or None")
    if gradient.shape != parameter.shape:
        raise ValueError(
            f"gradient {index} shape mismatch: expected {parameter.shape}, got {gradient.shape}"
        )
    if not np.issubdtype(gradient.dtype, np.floating):
        raise TypeError(f"gradient {index} must contain floating-point values")
    if not np.isfinite(gradient).all():
        raise ValueError(f"gradient {index} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = np.asarray(gradient, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"gradient {index} must fit in float64")
    return np.array(converted, dtype=np.float64, copy=True)


def _task_scale(task):
    maximum = 0.0
    for value in task:
        if value.size == 0:
            continue
        local = float(np.max(np.abs(value)))
        if local > maximum:
            maximum = local
    return maximum


def _scaled_squared_norm(task, scale):
    if scale == 0.0:
        return 0.0
    parts = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for value in task:
                scaled = value / scale
                parts.append(float(np.sum(scaled * scaled, dtype=np.float64)))
    except FloatingPointError as exc:
        raise ValueError("scaled gradient norm must remain finite") from exc
    total = math.fsum(parts)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("scaled gradient norm must remain positive and finite")
    return total


def _reported_norm(scale, scaled_squared_norm):
    if scale == 0.0:
        return 0.0, False
    unit_norm = math.sqrt(scaled_squared_norm)
    try:
        value = scale * unit_norm
    except OverflowError:
        return None, True
    if not math.isfinite(value):
        return None, True
    return float(value), False


def _cosine(left, right, left_scale, right_scale, left_sq, right_sq):
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    parts = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for left_value, right_value in zip(left, right):
                left_scaled = left_value / left_scale
                right_scaled = right_value / right_scale
                parts.append(
                    float(np.sum(left_scaled * right_scaled, dtype=np.float64))
                )
    except FloatingPointError as exc:
        raise ValueError("scaled gradient inner product must remain finite") from exc
    inner = math.fsum(parts)
    denominator = math.sqrt(left_sq) * math.sqrt(right_sq)
    if not math.isfinite(inner) or not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("scaled gradient cosine must remain finite")
    value = inner / denominator
    if not math.isfinite(value):
        raise ValueError("gradient cosine must remain finite")
    return float(max(-1.0, min(1.0, value)))


class GradientConflictAnalyzer:
    """Capture task gradients and report pairwise cosine conflicts.

    Each :meth:`capture` snapshots the current live gradient collection as one
    named task. Missing gradients are exact zero contributions. Analysis is
    observational: it never changes Tensor data, gradient buffers, mutation
    versions, or NumPy RNG state.
    """

    def __init__(self, parameters):
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        self._tasks = []
        self._names = []
        self._next_auto_index = 0
        self._lock = threading.RLock()

    @property
    def parameters(self):
        return self._parameters

    @property
    def task_count(self):
        with self._lock:
            return len(self._tasks)

    def _validate_binding(self):
        for index, (parameter, shape) in enumerate(zip(self._parameters, self._shapes)):
            if parameter.shape != shape:
                raise ValueError(
                    f"parameter {index} shape changed: expected {shape}, got {parameter.shape}"
                )
            if not parameter.requires_grad:
                raise ValueError(f"parameter {index} no longer requires gradients")

    def _resolve_name(self, name):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("task name must be a string or None")
            if not name:
                raise ValueError("task name must not be empty")
            if name in self._names:
                raise ValueError(f"duplicate task name: {name}")
            return name, self._next_auto_index

        index = self._next_auto_index
        while True:
            candidate = f"task_{index}"
            index += 1
            if candidate not in self._names:
                return candidate, index

    def capture(self, name=None):
        """Snapshot current live gradients as one task and return its name."""
        with self._lock:
            self._validate_binding()
            resolved_name, next_auto_index = self._resolve_name(name)
            task = tuple(
                _gradient_snapshot(parameter, index)
                for index, parameter in enumerate(self._parameters)
            )
            self._tasks.append(task)
            self._names.append(resolved_name)
            self._next_auto_index = next_auto_index
            return resolved_name

    def task_gradients(self):
        """Return independent copies of captured named task gradients."""
        with self._lock:
            return tuple(
                (
                    name,
                    tuple(np.array(value, dtype=np.float64, copy=True) for value in task),
                )
                for name, task in zip(self._names, self._tasks)
            )

    def report(self):
        """Return a deterministic strict-JSON-safe gradient conflict report."""
        with self._lock:
            self._validate_binding()
            task_count = len(self._tasks)
            scales = tuple(_task_scale(task) for task in self._tasks)
            squared_norms = tuple(
                _scaled_squared_norm(task, scale)
                for task, scale in zip(self._tasks, scales)
            )
            reported = tuple(
                _reported_norm(scale, squared)
                for scale, squared in zip(scales, squared_norms)
            )

            cosine_matrix = [[None for _ in range(task_count)] for _ in range(task_count)]
            comparable_matrix = [[False for _ in range(task_count)] for _ in range(task_count)]
            conflict_matrix = [[False for _ in range(task_count)] for _ in range(task_count)]
            per_task_comparable = [0 for _ in range(task_count)]
            per_task_conflicts = [0 for _ in range(task_count)]
            pairs = []
            pair_cosines = []
            conflict_pair_count = 0

            for index in range(task_count):
                if scales[index] != 0.0:
                    cosine_matrix[index][index] = 1.0
                    comparable_matrix[index][index] = True

            for left in range(task_count):
                for right in range(left + 1, task_count):
                    cosine = _cosine(
                        self._tasks[left],
                        self._tasks[right],
                        scales[left],
                        scales[right],
                        squared_norms[left],
                        squared_norms[right],
                    )
                    if cosine is None:
                        status = "undefined"
                        conflict = False
                    else:
                        cosine_matrix[left][right] = cosine
                        cosine_matrix[right][left] = cosine
                        comparable_matrix[left][right] = True
                        comparable_matrix[right][left] = True
                        per_task_comparable[left] += 1
                        per_task_comparable[right] += 1
                        pair_cosines.append(cosine)
                        conflict = cosine < 0.0
                        if conflict:
                            status = "conflict"
                            conflict_pair_count += 1
                            per_task_conflicts[left] += 1
                            per_task_conflicts[right] += 1
                            conflict_matrix[left][right] = True
                            conflict_matrix[right][left] = True
                        elif cosine > 0.0:
                            status = "aligned"
                        else:
                            status = "orthogonal"
                    pairs.append(
                        {
                            "left": self._names[left],
                            "right": self._names[right],
                            "cosine": cosine,
                            "status": status,
                            "conflict": bool(conflict),
                        }
                    )

            comparable_pair_count = len(pair_cosines)
            if pair_cosines:
                mean_cosine = float(math.fsum(pair_cosines) / comparable_pair_count)
                min_cosine = float(min(pair_cosines))
                max_cosine = float(max(pair_cosines))
                conflict_fraction = float(conflict_pair_count / comparable_pair_count)
            else:
                mean_cosine = None
                min_cosine = None
                max_cosine = None
                conflict_fraction = None

            per_task = []
            for index, name in enumerate(self._names):
                norm, norm_overflow = reported[index]
                comparable = per_task_comparable[index]
                conflicts = per_task_conflicts[index]
                per_task.append(
                    {
                        "name": name,
                        "l2_norm": norm,
                        "l2_overflow": bool(norm_overflow),
                        "zero_gradient": scales[index] == 0.0,
                        "comparable_peer_count": comparable,
                        "conflict_peer_count": conflicts,
                        "conflict_fraction": (
                            None if comparable == 0 else float(conflicts / comparable)
                        ),
                    }
                )

            return {
                "task_count": task_count,
                "parameter_count": len(self._parameters),
                "task_names": list(self._names),
                "cosine_similarity_matrix": cosine_matrix,
                "comparable_matrix": comparable_matrix,
                "conflict_matrix": conflict_matrix,
                "pair_count": task_count * (task_count - 1) // 2,
                "comparable_pair_count": comparable_pair_count,
                "conflict_pair_count": conflict_pair_count,
                "conflict_fraction": conflict_fraction,
                "mean_pair_cosine": mean_cosine,
                "min_pair_cosine": min_cosine,
                "max_pair_cosine": max_cosine,
                "pairs": pairs,
                "tasks": per_task,
            }

    def reset(self):
        """Discard captured task snapshots without touching live gradients."""
        with self._lock:
            self._tasks = []
            self._names = []
            self._next_auto_index = 0
            return self
