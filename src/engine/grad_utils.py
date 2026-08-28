"""Reusable global gradient norm measurement and clipping helpers."""

import numpy as np

from .tensor import Tensor


def _validate_max_norm(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("max_norm must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("max_norm must be finite") from exc
    if not np.isfinite(value):
        raise ValueError("max_norm must be finite")
    if value < 0.0:
        raise ValueError("max_norm must be non-negative")
    return value


def _collect_gradients(parameters):
    try:
        parameters = tuple(parameters)
    except TypeError as exc:
        raise TypeError("parameters must be an iterable of Tensors") from exc

    gradients = []
    seen = set()
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("parameters must not contain duplicate Tensor references")
        seen.add(marker)

        grad = parameter.grad
        if grad is None:
            continue
        if not isinstance(grad, np.ndarray):
            raise TypeError(f"gradient {index} must be a NumPy array")
        if not np.issubdtype(grad.dtype, np.floating):
            raise TypeError(f"gradient {index} must have a real floating dtype")
        if grad.shape != parameter.shape:
            raise ValueError(
                f"gradient {index} shape mismatch: expected {parameter.shape}, "
                f"got {grad.shape}"
            )
        if not np.isfinite(grad).all():
            raise ValueError(f"gradient {index} must contain only finite values")
        gradients.append((index, grad))
    return gradients


def _norm_parts(gradients):
    largest = 0.0
    for _, grad in gradients:
        if grad.size:
            largest = max(largest, float(np.max(np.abs(grad))))
    if largest == 0.0:
        return 0.0, 0.0, 0.0

    scaled_sumsq = 0.0
    for _, grad in gradients:
        scaled = np.asarray(grad, dtype=np.float64) / largest
        scaled_sumsq += float(np.sum(scaled * scaled, dtype=np.float64))
    scaled_norm = float(np.sqrt(scaled_sumsq))

    float_max = np.finfo(np.float64).max
    if scaled_norm > 0.0 and largest > float_max / scaled_norm:
        total = float("inf")
    else:
        total = largest * scaled_norm
    return largest, scaled_norm, float(total)


def global_grad_norm(parameters):
    """Return the global L2 norm of all present finite Tensor gradients."""
    gradients = _collect_gradients(parameters)
    _, _, total = _norm_parts(gradients)
    return total


def clip_grad_norm_(parameters, max_norm=1.0):
    """Return the global L2 norm and clip present gradients in place.

    ``max_norm=0`` follows the trainer convention and disables mutation while
    still measuring the norm. All parameter and gradient validation finishes
    before any gradient buffer is modified.
    """
    max_norm = _validate_max_norm(max_norm)
    gradients = _collect_gradients(parameters)
    largest, scaled_norm, total = _norm_parts(gradients)

    if max_norm == 0.0 or largest == 0.0:
        return total
    if np.isfinite(total) and total <= max_norm:
        return total

    for index, grad in gradients:
        if not grad.flags.writeable:
            raise ValueError(f"gradient {index} must be writeable for clipping")

    scale = (max_norm / largest) / scaled_norm
    # Clipping can legitimately round tiny components to zero. That is not a
    # failed transaction and should remain valid under caller warning policies.
    with np.errstate(under="ignore"):
        for _, grad in gradients:
            grad *= scale
    return total
