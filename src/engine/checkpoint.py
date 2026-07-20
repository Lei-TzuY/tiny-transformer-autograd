"""Checkpoint persistence for models, optimizers, and schedulers."""

import os
import pickle


def read_checkpoint(path):
    """Read a trusted local checkpoint file."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_checkpoint(path, model, optimizer=None, scheduler=None, step=0, metadata=None):
    """Save training state atomically."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("step", 0)
