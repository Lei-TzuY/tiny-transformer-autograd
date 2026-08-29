"""Safe persistent trainability transitions for leaf Tensors."""

from collections.abc import Iterable
from numbers import Integral
import threading

import numpy as np

from .tensor import Tensor


_TRAINABILITY_LOCK = threading.RLock()


def _normalize_enabled(enabled):
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be boolean")
    return bool(enabled)


def _materialize_parameters(parameters):
    if isinstance(parameters, Tensor):
        values = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("parameters must be a Tensor or iterable of Tensors")
        values = tuple(parameters)

    seen = set()
    records = []
    for index, parameter in enumerate(values):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError(
                "parameters must not contain duplicate Tensor references: "
                f"duplicate at index {index}"
            )
        seen.add(marker)

        if getattr(parameter, "_children", ()):
            raise ValueError(f"parameter {index} must be a leaf Tensor")

        current = parameter.requires_grad
        if not isinstance(current, (bool, np.bool_)):
            raise TypeError(f"parameter {index} requires_grad must be boolean")
        current = bool(current)

        version = getattr(parameter, "_version", None)
        if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral):
            raise TypeError(f"parameter {index} mutation version must be an integer")
        version = int(version)
        if version < 0:
            raise ValueError(
                f"parameter {index} mutation version must be non-negative"
            )

        shape = parameter.shape
        namespace = vars(parameter)
        records.append((parameter, namespace, current, version, shape))
    return tuple(records)


def set_trainable_(parameters, enabled):
    """Persistently set leaf Tensor trainability and return transition count.

    A true trainability transition clears the Tensor's live gradient slot and
    increments its mutation version.  The version bump intentionally makes any
    graph built before the transition stale, preventing backward closures from
    silently observing a different ``requires_grad`` value than the forward
    pass.  Disabling trainability also clears stale gradients even when the
    Tensor was already frozen, so optimizers that were bound earlier skip it.

    Call this helper only at a graph boundary: consume or discard current
    forward graphs before changing trainability, then build a fresh forward
    pass afterwards.
    """
    enabled = _normalize_enabled(enabled)

    with _TRAINABILITY_LOCK:
        records = _materialize_parameters(parameters)
        transition_count = sum(current != enabled for _, _, current, _, _ in records)

        # Every failure-prone public validation and shape lookup above finishes
        # before the first state write. Tensor stores these fields as ordinary
        # instance attributes, so the commit below is a compact all-or-nothing
        # helper-managed transaction under the process-local lock.
        for _parameter, namespace, current, version, shape in records:
            if current == enabled:
                if not enabled:
                    # Frozen tensors must never carry an active stale gradient:
                    # built-in optimizers key their skip decision on grad=None.
                    namespace["grad"] = None
                    namespace["_grad_shape"] = shape
                continue

            namespace["requires_grad"] = enabled
            namespace["grad"] = None
            namespace["_grad_shape"] = shape
            namespace["_version"] = version + 1

        return transition_count


def freeze_(parameters):
    """Persistently disable gradients for leaf Tensors."""
    return set_trainable_(parameters, False)


def unfreeze_(parameters):
    """Persistently enable gradients for leaf Tensors."""
    return set_trainable_(parameters, True)
