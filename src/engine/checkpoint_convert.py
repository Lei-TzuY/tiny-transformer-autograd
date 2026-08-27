"""One-way migration from trusted pickle checkpoints to the safe format.

The historical checkpoint reader necessarily executes pickle and must therefore
only be used with trusted local files.  This helper is intentionally one-way:
once the trusted state has been decoded and envelope-validated, it is re-encoded
through the repository's non-executable NPZ/JSON safe-checkpoint writer without
constructing a model or mutating process RNG/model/optimizer state.
"""

import os

from .checkpoint import read_checkpoint
from .safe_checkpoint import _write_safe_state


def convert_checkpoint_to_safe(source, destination):
    """Convert one trusted pickle checkpoint into a safe checkpoint.

    Parameters
    ----------
    source : path-like
        Existing trusted pickle checkpoint. Reading pickle can execute code, so
        this function must never be used on untrusted input.
    destination : path-like
        Safe NPZ/JSON checkpoint path. The destination is replaced atomically
        only after the trusted source has been fully read and validated.

    Returns
    -------
    int
        The completed training step stored in the checkpoint.

    Notes
    -----
    No model, optimizer, scheduler, or global NumPy RNG state is restored during
    conversion. Unsupported values in the historical state are rejected by the
    safe encoder rather than silently coerced.
    """
    source = os.fspath(source)
    destination = os.fspath(destination)
    state = read_checkpoint(source)
    _write_safe_state(destination, state)
    return state.get("step", 0)
