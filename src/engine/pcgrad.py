"""Projected conflicting-gradient (PCGrad) utilities for multi-task training."""

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


def _seed_value(seed):
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    value = int(seed)
    if value < 0 or value > 2**64 - 1:
        raise ValueError("seed must be in [0, 2**64 - 1]")
    return value


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
        with np.errstate(over="ignore", invalid="ignore"):
            local = float(np.max(np.abs(value)))
        if local > maximum:
            maximum = local
    return maximum


def _scaled_inner_and_reference_norm(current, reference):
    current_scale = _task_scale(current)
    reference_scale = _task_scale(reference)
    if current_scale == 0.0 or reference_scale == 0.0:
        return 0.0, 0.0, current_scale, reference_scale

    inner_parts = []
    norm_parts = []
    with np.errstate(over="raise", invalid="raise", under="ignore", divide="raise"):
        for left, right in zip(current, reference):
            left_scaled = left / current_scale
            right_scaled = right / reference_scale
            inner_parts.append(float(np.sum(left_scaled * right_scaled, dtype=np.float64)))
            norm_parts.append(float(np.sum(right_scaled * right_scaled, dtype=np.float64)))

    inner = math.fsum(inner_parts)
    norm = math.fsum(norm_parts)
    if not math.isfinite(inner) or not math.isfinite(norm):
        raise ValueError("scaled PCGrad inner products must remain finite")
    return inner, norm, current_scale, reference_scale


def _project_against(current, reference):
    inner, reference_norm, current_scale, _ = _scaled_inner_and_reference_norm(
        current, reference
    )
    if inner >= 0.0 or current_scale == 0.0 or reference_norm == 0.0:
        return tuple(np.array(value, copy=True) for value in current)

    ratio = inner / reference_norm
    candidates = []
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore", divide="raise"):
            reference_scale = _task_scale(reference)
            for left, right in zip(current, reference):
                left_scaled = left / current_scale
                right_scaled = right / reference_scale
                normalized = left_scaled - ratio * right_scaled
                candidate = normalized * current_scale
                if not np.isfinite(candidate).all():
                    raise ValueError("projected gradients must remain finite")
                candidates.append(np.array(candidate, dtype=np.float64, copy=True))
    except FloatingPointError as exc:
        raise ValueError("projected gradients must remain finite") from exc
    return tuple(candidates)


def _mean_tasks(tasks, shapes):
    if not tasks:
        raise RuntimeError("no task gradients have been captured")
    mean = tuple(np.zeros(shape, dtype=np.float64) for shape in shapes)
    for index, task in enumerate(tasks, start=1):
        if index == 1:
            mean = tuple(np.array(value, copy=True) for value in task)
            continue
        previous_weight = (index - 1) / index
        incoming_weight = 1.0 / index
        candidates = []
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                for previous, incoming in zip(mean, task):
                    candidate = previous * previous_weight + incoming * incoming_weight
                    if not np.isfinite(candidate).all():
                        raise ValueError("combined PCGrad gradients must remain finite")
                    candidates.append(np.array(candidate, dtype=np.float64, copy=True))
        except FloatingPointError as exc:
            raise ValueError("combined PCGrad gradients must remain finite") from exc
        mean = tuple(candidates)
    return mean


class PCGradBuffer:
    """Capture task gradients, project conflicts, and emit one combined gradient.

    Each :meth:`capture` snapshots the current live ``Tensor.grad`` values as one
    task. Missing gradients contribute exact zeros. :meth:`projected_gradients`
    implements PCGrad by projecting each task gradient against every other task
    whose inner product is negative, then returns the mean projected gradient.

    The default projection order is deterministic. Passing ``seed`` requests
    reproducible randomized peer order using an isolated NumPy Generator, so
    the process-global NumPy RNG is never consumed.
    """

    def __init__(self, parameters):
        self._parameters = _materialize_parameters(parameters)
        self._shapes = tuple(parameter.shape for parameter in self._parameters)
        self._tasks = []
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

    def capture(self):
        """Snapshot the current live gradients as one independent task."""
        with self._lock:
            self._validate_binding()
            task = tuple(
                _gradient_snapshot(parameter, index)
                for index, parameter in enumerate(self._parameters)
            )
            self._tasks.append(task)
            return len(self._tasks)

    def task_gradients(self):
        """Return independent copies of every captured task gradient."""
        with self._lock:
            return tuple(
                tuple(np.array(value, copy=True) for value in task)
                for task in self._tasks
            )

    def projected_task_gradients(self, *, seed=None):
        """Return independent PCGrad-projected gradients for each captured task."""
        seed = _seed_value(seed)
        with self._lock:
            if not self._tasks:
                raise RuntimeError("no task gradients have been captured")
            self._validate_binding()
            originals = tuple(
                tuple(np.array(value, copy=True) for value in task)
                for task in self._tasks
            )
            rng = None if seed is None else np.random.default_rng(seed)
            projected = []
            task_count = len(originals)
            for index, task in enumerate(originals):
                current = tuple(np.array(value, copy=True) for value in task)
                peers = [peer for peer in range(task_count) if peer != index]
                if rng is not None and len(peers) > 1:
                    peers = [int(value) for value in rng.permutation(peers)]
                for peer in peers:
                    current = _project_against(current, originals[peer])
                projected.append(current)
            return tuple(projected)

    def projected_gradients(self, *, seed=None):
        """Return the independent mean of all PCGrad-projected task gradients."""
        seed = _seed_value(seed)
        with self._lock:
            projected = self.projected_task_gradients(seed=seed)
            return tuple(np.array(value, copy=True) for value in _mean_tasks(projected, self._shapes))

    def copy_to_grads(self, *, seed=None):
        """Transactionally replace live ``.grad`` slots with projected gradients."""
        seed = _seed_value(seed)
        with self._lock:
            candidates = self.projected_gradients(seed=seed)
            previous = tuple(parameter.grad for parameter in self._parameters)
            attempted = 0
            try:
                for parameter, candidate in zip(self._parameters, candidates):
                    attempted += 1
                    parameter.grad = np.array(candidate, copy=True)
            except BaseException as exc:
                rollback_error = None
                for parameter, old in zip(
                    self._parameters[:attempted], previous[:attempted]
                ):
                    try:
                        parameter.grad = old
                    except BaseException as rollback_exc:
                        rollback_error = rollback_exc
                        break
                if rollback_error is not None:
                    raise RuntimeError("PCGrad gradient rollback failed") from rollback_error
                raise exc
            return self

    def reset(self):
        """Discard captured task gradients without touching live gradients."""
        with self._lock:
            self._tasks = []
            return self
