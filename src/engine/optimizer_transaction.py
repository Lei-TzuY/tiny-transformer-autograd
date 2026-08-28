"""Exception-safe transactions for built-in optimizer steps.

The helper in this module snapshots both model parameters and optimizer state,
then either commits a healthy step or rolls the complete optimizer/parameter
state back.  Rollback intentionally never rewinds Tensor mutation versions:
if a failed step wrote a Tensor, pre-existing autograd graphs must stay stale.
"""

from contextlib import contextmanager
import copy
import math
from numbers import Complex, Real
import threading
import weakref

import numpy as np

from .optim import Adam, AdamW, SGD
from .tensor import Tensor


_SUPPORTED_OPTIMIZERS = (SGD, Adam, AdamW)
_LOCKS_GUARD = threading.Lock()
_OPTIMIZER_LOCKS = weakref.WeakKeyDictionary()


def _optimizer_lock(optimizer):
    """Return a stable reentrant lock for one optimizer instance."""
    with _LOCKS_GUARD:
        lock = _OPTIMIZER_LOCKS.get(optimizer)
        if lock is None:
            lock = threading.RLock()
            _OPTIMIZER_LOCKS[optimizer] = lock
        return lock


def _normalise_optimizer(optimizer):
    if not isinstance(optimizer, _SUPPORTED_OPTIMIZERS):
        raise TypeError(
            "optimizer_step_transaction optimizer must be SGD, Adam, or AdamW"
        )
    parameters = getattr(optimizer, "parameters", None)
    if not isinstance(parameters, list):
        raise TypeError("optimizer parameters must be stored in a list")

    seen = set()
    materialized = []
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, Tensor):
            raise TypeError(f"optimizer parameter {index} must be a Tensor")
        marker = id(parameter)
        if marker in seen:
            raise ValueError("optimizer parameters must not contain duplicates")
        seen.add(marker)
        materialized.append(parameter)
    return parameters, tuple(materialized)


def _parameter_snapshot(parameter, index):
    data = np.asarray(parameter.data)
    if not np.isfinite(data).all():
        raise ValueError(
            f"optimizer transaction parameter {index} must contain only finite values"
        )
    return {
        "data": np.array(data, dtype=np.float64, copy=True),
        "requires_grad": bool(parameter.requires_grad),
    }


def _format_path(path, key):
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}[{key!r}]"


def _validate_finite_state(state):
    """Reject non-finite numeric optimizer state without mutating it."""
    stack = [("$", state)]
    seen_containers = set()
    while stack:
        path, value = stack.pop()

        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(
                    f"optimizer transaction state {path} must contain only finite values"
                )
            continue

        if isinstance(value, np.generic):
            if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value)):
                raise ValueError(
                    f"optimizer transaction state {path} must contain only finite values"
                )
            continue

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"optimizer transaction state {path} must contain only finite values"
                )
            continue
        if isinstance(value, complex):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError(
                    f"optimizer transaction state {path} must contain only finite values"
                )
            continue

        if isinstance(value, dict):
            marker = id(value)
            if marker in seen_containers:
                continue
            seen_containers.add(marker)
            items = list(value.items())
            for key, child in reversed(items):
                stack.append((_format_path(path, key), child))
            continue

        if isinstance(value, (list, tuple)):
            marker = id(value)
            if marker in seen_containers:
                continue
            seen_containers.add(marker)
            for index in range(len(value) - 1, -1, -1):
                stack.append((_format_path(path, index), value[index]))


def _snapshot_optimizer_state(optimizer):
    state = copy.deepcopy(optimizer.state_dict())
    _validate_finite_state(state)
    return state


def _same_parameter_values(parameter, snapshot):
    current = np.asarray(parameter.data)
    saved = snapshot["data"]
    return current.shape == saved.shape and np.array_equal(current, saved)


class _OptimizerTransaction:
    def __init__(self, optimizer):
        parameter_container, parameters = _normalise_optimizer(optimizer)
        snapshots = tuple(
            _parameter_snapshot(parameter, index)
            for index, parameter in enumerate(parameters)
        )
        optimizer_state = _snapshot_optimizer_state(optimizer)

        self.optimizer = optimizer
        self.parameter_container = parameter_container
        self.parameters = parameters
        self.snapshots = snapshots
        self.optimizer_state = optimizer_state

    def _validate_parameter_identity(self):
        current = getattr(self.optimizer, "parameters", None)
        if not isinstance(current, list):
            raise RuntimeError(
                "optimizer parameter collection changed during transaction"
            )
        if len(current) != len(self.parameters) or any(
            current_parameter is not saved_parameter
            for current_parameter, saved_parameter in zip(current, self.parameters)
        ):
            raise RuntimeError(
                "optimizer parameter collection changed during transaction"
            )

    def validate_commit(self):
        self._validate_parameter_identity()
        for index, (parameter, snapshot) in enumerate(
            zip(self.parameters, self.snapshots)
        ):
            current = np.asarray(parameter.data)
            expected_shape = snapshot["data"].shape
            if current.shape != expected_shape:
                raise ValueError(
                    "optimizer transaction parameter shape changed at index "
                    f"{index}: expected {expected_shape}, got {current.shape}"
                )
            if not np.isfinite(current).all():
                raise ValueError(
                    f"optimizer transaction parameter {index} became non-finite"
                )
            if bool(parameter.requires_grad) != snapshot["requires_grad"]:
                raise ValueError(
                    "optimizer transaction requires_grad changed at index "
                    f"{index}"
                )

        current_state = self.optimizer.state_dict()
        _validate_finite_state(current_state)

    def rollback(self):
        # Restore the built-in optimizer's original parameter list before its
        # load_state_dict() validates state buffer counts and shapes.
        if getattr(self.optimizer, "parameters", None) is not self.parameter_container:
            self.optimizer.parameters = self.parameter_container
        self.parameter_container[:] = self.parameters

        for parameter, snapshot in zip(self.parameters, self.snapshots):
            saved = snapshot["data"]
            if not _same_parameter_values(parameter, snapshot):
                current = np.asarray(parameter.data)
                if current.shape == saved.shape and parameter.data.flags.writeable:
                    parameter.data[...] = saved
                else:
                    # Replacing storage is the recovery path for shape drift or
                    # caller-made read-only storage.  Tensor.data tracks this as
                    # another mutation, which is required for graph safety.
                    parameter.data = saved
            parameter.requires_grad = snapshot["requires_grad"]

        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))


def _rollback_after_failure(transaction):
    try:
        transaction.rollback()
    except BaseException as rollback_error:
        raise RuntimeError("optimizer step transaction rollback failed") from rollback_error


@contextmanager
def optimizer_step_transaction(optimizer):
    """Make one built-in optimizer step exception/non-finite safe.

    Typical usage::

        with optimizer_step_transaction(optimizer):
            optimizer.step()

    The transaction snapshots every bound parameter plus the optimizer's own
    ``state_dict()``.  A body exception rolls both back and is re-raised.  On
    normal exit, parameter identity, shape, finiteness, ``requires_grad``, and
    optimizer-state finiteness are checked; a failed check is rolled back and
    then raised.

    Successful steps are left untouched.  Rollback restores parameter *values*
    but never decrements Tensor mutation versions, so graphs built before a
    failed write remain correctly invalidated.

    Transactions on the same optimizer instance are serialized across threads
    with a reentrant lock; same-thread nesting remains supported.  Direct calls
    to ``optimizer.step()`` outside this helper are not synchronized by it.

    The context currently supports the repository's built-in ``SGD``, ``Adam``,
    and ``AdamW`` optimizers.  Gradient buffers and NumPy RNG state are outside
    the transaction because these optimizer ``step()`` implementations do not
    mutate or consume them.
    """
    # Validate the public optimizer type before registering/acquiring a lock.
    if not isinstance(optimizer, _SUPPORTED_OPTIMIZERS):
        _normalise_optimizer(optimizer)

    lock = _optimizer_lock(optimizer)
    with lock:
        transaction = _OptimizerTransaction(optimizer)
        try:
            yield optimizer
        except BaseException:
            _rollback_after_failure(transaction)
            raise

        try:
            transaction.validate_commit()
        except BaseException:
            _rollback_after_failure(transaction)
            raise
