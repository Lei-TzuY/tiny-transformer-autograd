"""Lookahead optimizer wrapper for the repository's built-in optimizers.

Lookahead maintains a slow copy of each parameter while an inner optimizer
updates the live (fast) parameters.  Every ``sync_period`` successful inner
steps, slow weights move toward fast weights and the interpolated values are
copied back into the live Tensors.
"""

from copy import deepcopy
from numbers import Integral, Real
import threading

import numpy as np

from .optim import Adam, AdamW, SGD
from .tensor import Tensor


_SUPPORTED_OPTIMIZERS = (SGD, Adam, AdamW)
_STATE_VERSION = 1


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_interval(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return value


def _nonnegative_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(name, value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _snapshot_parameters(optimizer):
    if not isinstance(optimizer, _SUPPORTED_OPTIMIZERS):
        raise TypeError("Lookahead optimizer must be SGD, Adam, or AdamW")

    parameters = getattr(optimizer, "parameters", None)
    if not isinstance(parameters, list):
        raise TypeError("Lookahead optimizer parameters must be stored in a list")

    materialized = []
    seen = set()
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"Lookahead optimizer parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("Lookahead optimizer parameters must not contain duplicates")
        seen.add(marker)

        data = np.asarray(parameter.data)
        if not np.isfinite(data).all():
            raise ValueError(
                f"Lookahead optimizer parameter {index} must contain only finite values"
            )
        materialized.append(parameter)

    return parameters, tuple(materialized)


def _snapshot_slow(parameters):
    return [
        np.array(parameter.data, dtype=np.float64, copy=True, subok=False)
        for parameter in parameters
    ]


def _interpolate(slow, fast, alpha):
    """Return a warning-neutral convex interpolation of two finite arrays."""
    if alpha == 0.0:
        return slow.copy()
    if alpha == 1.0:
        return np.array(fast, dtype=np.float64, copy=True, subok=False)

    fast = np.asarray(fast, dtype=np.float64)
    result = np.empty_like(slow, dtype=np.float64)
    same_sign = np.signbit(slow) == np.signbit(fast)

    # For same-sign endpoints, fast - slow cannot exceed the binary64 range.
    # For opposite signs that subtraction can overflow, so use a convex weighted
    # sum whose two terms have opposite signs and individually shrink in magnitude.
    with np.errstate(over="raise", invalid="raise"):
        if np.any(same_sign):
            before = slow[same_sign]
            after = fast[same_sign]
            result[same_sign] = before + alpha * (after - before)
        opposite = ~same_sign
        if np.any(opposite):
            before = slow[opposite]
            after = fast[opposite]
            result[opposite] = (1.0 - alpha) * before + alpha * after

    if not np.isfinite(result).all():
        raise ValueError("Lookahead interpolation produced a non-finite value")
    return result


def _coerce_slow_weights(saved, parameters):
    if not isinstance(saved, (list, tuple)):
        raise TypeError("Lookahead slow_weights must be a list or tuple")
    if len(saved) != len(parameters):
        raise ValueError(
            "Lookahead slow weight count mismatch: "
            f"expected {len(parameters)}, got {len(saved)}"
        )

    snapshots = []
    for index, (value, parameter) in enumerate(zip(saved, parameters)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Lookahead slow_weights[{index}] must be a NumPy array")
        if value.shape != parameter.data.shape:
            raise ValueError(
                f"Lookahead slow_weights[{index}] shape mismatch: expected "
                f"{parameter.data.shape}, got {value.shape}"
            )
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype, np.complexfloating
        ):
            raise TypeError(
                f"Lookahead slow_weights[{index}] must have a real numeric dtype"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"Lookahead slow_weights[{index}] must contain only finite values"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.array(value, dtype=np.float64, copy=True, subok=False)
        if not np.isfinite(converted).all():
            raise ValueError(
                f"Lookahead slow_weights[{index}] must fit in float64"
            )
        snapshots.append(converted)
    return snapshots


class Lookahead:
    """Wrap ``SGD``, ``Adam``, or ``AdamW`` with Lookahead slow weights.

    Parameters
    ----------
    optimizer:
        A repository built-in optimizer whose live parameters are the fast weights.
    sync_period:
        Number of successful inner optimizer steps between automatic slow/fast
        synchronizations.
    alpha:
        Convex interpolation factor in ``[0, 1]``.  ``0`` restores the current
        slow weights at each synchronization; ``1`` makes the slow weights catch
        up to the fast weights without an additional Tensor write.

    Notes
    -----
    The wrapper does not make the *inner* optimizer step exception-atomic.  If
    ``optimizer.step()`` itself raises after partially mutating its own state,
    that failure belongs to the inner optimizer.  Slow weights and the Lookahead
    step counter advance only after the inner step returns successfully.
    """

    def __init__(self, optimizer, *, sync_period=5, alpha=0.5):
        sync_period = _positive_integer("Lookahead sync_period", sync_period)
        alpha = _unit_interval("Lookahead alpha", alpha)
        parameter_container, parameters = _snapshot_parameters(optimizer)

        self.optimizer = optimizer
        self._parameter_container = parameter_container
        self._parameters = parameters
        self._slow = _snapshot_slow(parameters)
        self._sync_period = sync_period
        self._alpha = alpha
        self._step_count = 0
        self._pending_sync = False
        self._lock = threading.RLock()

    @property
    def parameters(self):
        return self.optimizer.parameters

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        value = _unit_interval("Lookahead alpha", value)
        with self._lock:
            self._alpha = value

    @property
    def sync_period(self):
        return self._sync_period

    @sync_period.setter
    def sync_period(self, value):
        value = _positive_integer("Lookahead sync_period", value)
        with self._lock:
            self._sync_period = value

    @property
    def step_count(self):
        return self._step_count

    @property
    def pending_sync(self):
        return self._pending_sync

    def _validate_parameter_binding(self):
        current = getattr(self.optimizer, "parameters", None)
        if current is not self._parameter_container:
            raise RuntimeError("Lookahead optimizer parameter collection changed")
        if len(current) != len(self._parameters) or any(
            live is not saved for live, saved in zip(current, self._parameters)
        ):
            raise RuntimeError("Lookahead optimizer parameter collection changed")

        for index, (parameter, slow) in enumerate(zip(self._parameters, self._slow)):
            if parameter.data.shape != slow.shape:
                raise ValueError(
                    f"Lookahead parameter shape changed at index {index}: expected "
                    f"{slow.shape}, got {parameter.data.shape}"
                )

    def _synchronize_locked(self):
        self._validate_parameter_binding()

        new_slow = []
        changed = []
        for index, (parameter, slow) in enumerate(zip(self._parameters, self._slow)):
            fast = np.asarray(parameter.data)
            if not np.isfinite(fast).all():
                raise ValueError(
                    f"Lookahead fast parameter {index} must contain only finite values"
                )
            interpolated = _interpolate(slow, fast, self._alpha)
            new_slow.append(interpolated)
            needs_write = not np.array_equal(fast, interpolated)
            changed.append(needs_write)
            if needs_write and fast.size and not parameter.data.flags.writeable:
                raise ValueError(
                    f"Lookahead fast parameter {index} must be writeable for sync"
                )

        # Snapshot the post-inner-step fast values so an unexpected late write
        # failure cannot leave a partially synchronized parameter collection.
        fast_before_sync = [
            np.array(parameter.data, dtype=np.float64, copy=True, subok=False)
            for parameter in self._parameters
        ]
        written = []
        try:
            for index, (parameter, value, needs_write) in enumerate(
                zip(self._parameters, new_slow, changed)
            ):
                if not needs_write:
                    continue
                parameter.data[...] = value
                written.append(index)
        except BaseException:
            try:
                for index in written:
                    parameter = self._parameters[index]
                    previous = fast_before_sync[index]
                    if (
                        parameter.data.shape == previous.shape
                        and parameter.data.flags.writeable
                    ):
                        parameter.data[...] = previous
                    else:
                        parameter.data = previous
            except BaseException as rollback_error:
                raise RuntimeError("Lookahead synchronization rollback failed") from rollback_error
            raise

        self._slow = new_slow
        self._pending_sync = False

    def sync(self):
        """Synchronize slow and fast weights immediately.

        Manual synchronization does not change ``step_count`` or the automatic
        cadence.  It is also the recovery operation after a scheduled sync fails:
        ``pending_sync`` remains true until a later ``sync()`` succeeds.
        """
        with self._lock:
            self._synchronize_locked()
        return self

    def step(self):
        """Run one inner optimizer step and synchronize on the configured cadence."""
        with self._lock:
            if self._pending_sync:
                raise RuntimeError(
                    "Lookahead has a pending synchronization; call sync() before step()"
                )
            self._validate_parameter_binding()
            result = self.optimizer.step()
            self._step_count += 1
            if self._step_count % self._sync_period == 0:
                self._pending_sync = True
                self._synchronize_locked()
            return result

    def zero_grad(self, set_to_none=False):
        """Forward gradient clearing to the wrapped optimizer."""
        with self._lock:
            return self.optimizer.zero_grad(set_to_none=set_to_none)

    def slow_weights(self):
        """Return independent copies of the current slow weights."""
        with self._lock:
            self._validate_parameter_binding()
            return tuple(value.copy() for value in self._slow)

    def state_dict(self):
        """Return independent Lookahead and inner-optimizer state."""
        with self._lock:
            self._validate_parameter_binding()
            return {
                "version": _STATE_VERSION,
                "optimizer_type": type(self.optimizer).__name__,
                "sync_period": self._sync_period,
                "alpha": self._alpha,
                "step_count": self._step_count,
                "pending_sync": self._pending_sync,
                "slow_weights": [value.copy() for value in self._slow],
                "optimizer": deepcopy(self.optimizer.state_dict()),
            }

    def load_state_dict(self, state):
        """Transactionally restore Lookahead metadata, slow weights, and inner state."""
        if not isinstance(state, dict):
            raise TypeError("Lookahead state must be a dictionary")

        with self._lock:
            self._validate_parameter_binding()

            version = _nonnegative_integer("Lookahead state version", state["version"])
            if version != _STATE_VERSION:
                raise ValueError(
                    f"unsupported Lookahead state version {version}; expected {_STATE_VERSION}"
                )
            optimizer_type = state["optimizer_type"]
            if not isinstance(optimizer_type, str):
                raise TypeError("Lookahead optimizer_type must be a string")
            expected_type = type(self.optimizer).__name__
            if optimizer_type != expected_type:
                raise ValueError(
                    f"Lookahead optimizer type mismatch: expected {expected_type}, "
                    f"got {optimizer_type}"
                )

            sync_period = _positive_integer(
                "Lookahead sync_period", state["sync_period"]
            )
            alpha = _unit_interval("Lookahead alpha", state["alpha"])
            step_count = _nonnegative_integer(
                "Lookahead step_count", state["step_count"]
            )
            pending_sync = _boolean(
                "Lookahead pending_sync", state["pending_sync"]
            )
            slow = _coerce_slow_weights(state["slow_weights"], self._parameters)
            if "optimizer" not in state:
                raise KeyError("optimizer")
            inner_state = deepcopy(state["optimizer"])

            # Built-in optimizer loaders validate their state before normal commit.
            # Keep a rollback snapshot as an additional guard so a rejected inner
            # load cannot leave this wrapper observing partially changed state.
            inner_before = deepcopy(self.optimizer.state_dict())
            try:
                self.optimizer.load_state_dict(inner_state)
            except BaseException:
                try:
                    self.optimizer.load_state_dict(inner_before)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "Lookahead inner optimizer state rollback failed"
                    ) from rollback_error
                raise

            self._sync_period = sync_period
            self._alpha = alpha
            self._step_count = step_count
            self._pending_sync = pending_sync
            self._slow = slow
        return self
