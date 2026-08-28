"""Exponential moving averages for ordered Tensor parameter collections.

The helper is intentionally independent from the optimizer and trainer.  It
keeps ordinary NumPy shadow arrays, never participates in autograd, and only
mutates model Tensors when ``copy_to()`` or ``average_parameters()`` is used.
"""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from numbers import Integral, Real

import numpy as np

from .tensor import Tensor


def _normalise_decay(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("EMA decay must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("EMA decay must be finite") from exc
    if not np.isfinite(value):
        raise ValueError("EMA decay must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    return value


def _normalise_parameters(parameters):
    if isinstance(parameters, Tensor):
        values = (parameters,)
    else:
        if not isinstance(parameters, Iterable):
            raise TypeError("EMA parameters must be a Tensor or iterable of Tensors")
        values = tuple(parameters)
    if not values:
        raise ValueError("EMA parameters must contain at least one Tensor")

    seen = set()
    for index, value in enumerate(values):
        if not isinstance(value, Tensor):
            raise TypeError(f"EMA parameter {index} must be a Tensor")
        identity = id(value)
        if identity in seen:
            raise ValueError("EMA parameters must not contain duplicate Tensors")
        seen.add(identity)
    return values


def _finite_parameter_copy(parameter, index):
    data = np.asarray(parameter.data)
    if not np.isfinite(data).all():
        raise ValueError(f"EMA parameter {index} must contain only finite values")
    return np.array(data, dtype=np.float64, copy=True)


def _normalise_num_updates(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("EMA num_updates must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("EMA num_updates must be non-negative")
    return value


def _normalise_average(value, shape, index):
    raw = np.asarray(value)
    is_integer = np.issubdtype(raw.dtype, np.integer)
    is_floating = np.issubdtype(raw.dtype, np.floating)
    if np.issubdtype(raw.dtype, np.bool_) or not (is_integer or is_floating):
        raise TypeError(f"EMA average {index} must contain real numeric values")
    if raw.shape != shape:
        raise ValueError(
            f"EMA average {index} shape mismatch: expected {shape}, got {raw.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError(f"EMA average {index} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"EMA average {index} must fit in float64")
    return np.array(converted, copy=True)


class ExponentialMovingAverage:
    """Track an exponential moving average of a fixed ordered Tensor set.

    Shadows start as exact copies of the parameters.  Calling ``update()``
    applies ``shadow = decay * shadow + (1 - decay) * parameter`` and increments
    ``num_updates``.  Updating shadow state is read-only with respect to the
    bound Tensors and their gradient buffers.

    ``copy_to()`` copies the current shadows into the bound Tensors.  Such
    writes go through ``Tensor.data`` mutation tracking, so already-built graphs
    are invalidated normally.  ``average_parameters()`` temporarily installs
    the shadows and restores the exact entry values even if the body raises.
    """

    def __init__(self, parameters, decay=0.999):
        # Validate scalar options before consuming a caller-owned generator.
        decay = _normalise_decay(decay)
        parameters = _normalise_parameters(parameters)

        averages = []
        shapes = []
        for index, parameter in enumerate(parameters):
            averages.append(_finite_parameter_copy(parameter, index))
            shapes.append(parameter.shape)

        self._parameters = parameters
        self._shapes = tuple(shapes)
        self._averages = tuple(averages)
        self.decay = decay
        self.num_updates = 0

    def averages(self):
        """Return independent copies of the current shadow arrays."""
        return tuple(np.array(value, copy=True) for value in self._averages)

    def _validate_shapes(self):
        for index, (parameter, expected) in enumerate(
            zip(self._parameters, self._shapes)
        ):
            if parameter.shape != expected:
                raise ValueError(
                    f"EMA parameter {index} shape changed: expected {expected}, "
                    f"got {parameter.shape}"
                )

    def update(self):
        """Update every shadow from the current bound parameter values."""
        self._validate_shapes()
        current = [
            _finite_parameter_copy(parameter, index)
            for index, parameter in enumerate(self._parameters)
        ]

        if self.decay == 1.0:
            self.num_updates += 1
            return self
        if self.decay == 0.0:
            self._averages = tuple(current)
            self.num_updates += 1
            return self

        retain = self.decay
        add = 1.0 - retain
        updated = []
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                for average, value in zip(self._averages, current):
                    next_average = average * retain + value * add
                    if not np.isfinite(next_average).all():
                        raise FloatingPointError
                    updated.append(np.array(next_average, copy=True))
        except FloatingPointError as exc:
            # A convex combination of finite float64 values should remain
            # finite.  Fail before committing any shadow if NumPy reports
            # otherwise, rather than leaving a partially advanced EMA.
            raise ValueError("EMA update produced non-finite values") from exc

        self._averages = tuple(updated)
        self.num_updates += 1
        return self

    def copy_to(self):
        """Copy shadows into the bound Tensors after a complete preflight."""
        self._validate_shapes()
        writes = []
        for index, (parameter, average) in enumerate(
            zip(self._parameters, self._averages)
        ):
            data = np.asarray(parameter.data)
            if np.array_equal(data, average):
                continue
            if not parameter.data.flags.writeable:
                raise ValueError(f"EMA parameter {index} data is not writeable")
            writes.append((parameter, average))

        for parameter, average in writes:
            parameter.data[...] = average
        return self

    @contextmanager
    def average_parameters(self):
        """Temporarily install EMA values and restore entry values on exit."""
        self._validate_shapes()
        originals = tuple(
            np.array(parameter.data, copy=True) for parameter in self._parameters
        )
        self.copy_to()
        try:
            yield self
        finally:
            for parameter, original in zip(self._parameters, originals):
                data = np.asarray(parameter.data)
                if data.shape == original.shape and np.array_equal(
                    data, original, equal_nan=True
                ):
                    continue
                if data.shape == original.shape and parameter.data.flags.writeable:
                    parameter.data[...] = original
                else:
                    # The context body may have replaced storage or its shape.
                    # The property setter restores the original shape/value and
                    # deliberately records another Tensor mutation.
                    parameter.data = original

    def state_dict(self):
        """Return an independent, serializable snapshot of EMA state."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "averages": [np.array(value, copy=True) for value in self._averages],
        }

    def load_state_dict(self, state):
        """Transactionally restore EMA metadata and shadows for this shape set."""
        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        required = {"decay", "num_updates", "averages"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"EMA state is missing required keys: {missing}")

        decay = _normalise_decay(state["decay"])
        num_updates = _normalise_num_updates(state["num_updates"])
        averages = state["averages"]
        if not isinstance(averages, (list, tuple)):
            raise TypeError("EMA averages must be a list or tuple")
        if len(averages) != len(self._parameters):
            raise ValueError(
                "EMA average count mismatch: expected "
                f"{len(self._parameters)}, got {len(averages)}"
            )

        restored = tuple(
            _normalise_average(value, shape, index)
            for index, (value, shape) in enumerate(zip(averages, self._shapes))
        )

        self.decay = decay
        self.num_updates = num_updates
        self._averages = restored
        return self
