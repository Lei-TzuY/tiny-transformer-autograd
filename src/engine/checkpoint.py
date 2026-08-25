"""Checkpoint persistence for models, optimizers, and schedulers."""

from collections.abc import Mapping
from numbers import Integral
import os
import pickle

import numpy as np


CHECKPOINT_VERSION = 2


def read_checkpoint(path):
    """Read a trusted local checkpoint file."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_checkpoint(path, model, optimizer=None, scheduler=None, step=0, metadata=None):
    """Save training state atomically."""
    step = _nonnegative_checkpoint_step(step)
    state = {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_type": optimizer.__class__.__name__ if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": np.random.get_state(),
        "step": step,
        "metadata": metadata or {},
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def restore_checkpoint(state, model, optimizer=None, scheduler=None, strict=True):
    """Restore an already-read checkpoint and return its completed step."""
    step = _validate_checkpoint_envelope(state, strict=strict)

    if optimizer is not None and state.get("optimizer") is not None:
        saved_type = state.get("optimizer_type")
        current_type = optimizer.__class__.__name__
        if saved_type is not None and saved_type != current_type:
            raise ValueError(
                f"optimizer type mismatch: checkpoint uses {saved_type}, "
                f"but received {current_type}"
            )

    restore_optimizer = optimizer is not None and state.get("optimizer") is not None
    restore_scheduler = scheduler is not None and state.get("scheduler") is not None
    model_before = model.state_dict()
    optimizer_before = optimizer.state_dict() if restore_optimizer else None
    scheduler_before = scheduler.state_dict() if restore_scheduler else None
    rng_before = np.random.get_state()

    try:
        model.load_state_dict(state["model"], strict=strict)
        if restore_optimizer:
            optimizer.load_state_dict(state["optimizer"])
        if restore_scheduler:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("rng_state") is not None:
            np.random.set_state(state["rng_state"])
    except Exception:
        # Restore is transactional across all caller-owned state. This also
        # protects custom modules/optimizers whose own loaders are not atomic.
        model.load_state_dict(model_before, strict=True)
        if restore_optimizer:
            optimizer.load_state_dict(optimizer_before)
        if restore_scheduler:
            scheduler.load_state_dict(scheduler_before)
        np.random.set_state(rng_before)
        raise
    return step


def _validate_checkpoint_envelope(state, *, strict):
    """Validate restore metadata before snapshotting or mutating caller state."""
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a mapping")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")

    version = state.get("format_version", 1)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral):
        raise TypeError("checkpoint format_version must be an integer")
    version = int(version)
    if not 1 <= version <= CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {version}")

    if "model" not in state:
        raise ValueError("checkpoint is missing required model state")

    step = _nonnegative_checkpoint_step(state.get("step", 0))

    optimizer_type = state.get("optimizer_type")
    if optimizer_type is not None and (
        not isinstance(optimizer_type, str) or not optimizer_type
    ):
        raise TypeError("checkpoint optimizer_type must be a non-empty string or None")

    rng_state = state.get("rng_state")
    if rng_state is not None:
        # Validate against an isolated generator so malformed checkpoint data
        # cannot perturb the process-global NumPy RNG merely by being checked.
        probe = np.random.RandomState()
        try:
            probe.set_state(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint NumPy RNG state") from exc

    return step


def _nonnegative_checkpoint_step(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("checkpoint step must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return value
