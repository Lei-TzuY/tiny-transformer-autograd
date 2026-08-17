"""Checkpoint persistence for models, optimizers, and schedulers."""

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
    version = state.get("format_version", 1)
    if not isinstance(version, int) or not 1 <= version <= CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {version}")

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
    return state.get("step", 0)
