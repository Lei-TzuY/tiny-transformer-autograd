"""Transactional elementwise gradient clipping for Tensor collections.

The helper clips existing gradient buffers in place without replacing their ndarray
objects. The complete request is validated before the first write, and unexpected
commit-time failures roll every attempted buffer back to its exact entry values.
"""

from numbers import Real

import numpy as np

from .tensor import Tensor


def _positive_finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        items = [parameters]
    else:
        try:
            items = list(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be a Tensor or iterable of Tensors") from exc

    seen = set()
    for index, parameter in enumerate(items):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameters[{index}] must be a Tensor")
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise ValueError("parameters must not contain duplicate Tensor references")
        seen.add(parameter_id)
    return items


def _reject_overlapping_gradient_storage(gradient, previous_gradients, index):
    for previous in previous_gradients:
        if gradient is previous or np.shares_memory(gradient, previous):
            raise ValueError(
                f"gradient for parameters[{index}] must not share storage with another gradient"
            )


def _reject_parameter_storage_alias(gradient, parameters, index):
    for parameter in parameters:
        if np.shares_memory(gradient, parameter.data):
            raise ValueError(
                f"gradient for parameters[{index}] must not share storage with parameter data"
            )


def clip_grad_value_(parameters, clip_value):
    """Clamp present gradients to ``[-clip_value, clip_value]`` transactionally.

    Parameters with ``grad is None`` are ignored. Present gradients must be floating
    NumPy arrays with exactly the parameter shape and only finite values. Gradient
    buffers for distinct parameters must not overlap in storage. A gradient that needs
    clipping must not overlap any bound parameter's data storage. The function returns
    the number of gradient buffers whose values changed.

    Validation is all-or-nothing: malformed, aliased, or required read-only buffers are
    rejected before any gradient changes. If a later in-place assignment unexpectedly
    raises after mutating its destination, every attempted buffer is restored before
    re-raising.
    """

    limit = _positive_finite_real(clip_value, "clip_value")
    items = _materialize_parameters(parameters)

    active = []
    previous_gradients = []
    for index, parameter in enumerate(items):
        gradient = parameter.grad
        if gradient is None:
            continue
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"gradient for parameters[{index}] must be a NumPy array")
        if not np.issubdtype(gradient.dtype, np.floating):
            raise TypeError(f"gradient for parameters[{index}] must have a floating dtype")
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient for parameters[{index}] must match parameter shape")
        if not np.all(np.isfinite(gradient)):
            raise ValueError(f"gradient for parameters[{index}] must contain only finite values")

        _reject_overlapping_gradient_storage(gradient, previous_gradients, index)
        previous_gradients.append(gradient)

        candidate = np.clip(np.asarray(gradient), -limit, limit)
        changed = not np.array_equal(candidate, np.asarray(gradient))
        if changed:
            _reject_parameter_storage_alias(gradient, items, index)
            if not gradient.flags.writeable:
                raise ValueError(f"gradient for parameters[{index}] must be writeable")
        active.append((gradient, candidate, changed))

    originals = [np.array(gradient, copy=True) for gradient, _, _ in active]
    attempted = []
    try:
        for position, (gradient, candidate, changed) in enumerate(active):
            if not changed:
                continue
            attempted.append(position)
            gradient[...] = candidate
    except BaseException:
        try:
            for position in reversed(attempted):
                gradient = active[position][0]
                gradient[...] = originals[position]
        except BaseException as rollback_error:
            raise RuntimeError("gradient value clipping rollback failed") from rollback_error
        raise

    return sum(1 for _, _, changed in active if changed)
