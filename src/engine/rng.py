"""Exception-safe isolation for the process-global NumPy random state."""

from contextlib import contextmanager
import threading

import numpy as np


_MAX_LEGACY_SEED = 2**32 - 1
_RNG_FORK_LOCK = threading.RLock()


def _validate_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    seed = int(seed)
    if seed < 0 or seed > _MAX_LEGACY_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_LEGACY_SEED}]")
    return seed


def _validate_state(state):
    if state is None:
        return None
    probe = np.random.RandomState()
    try:
        probe.set_state(state)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("state must be a valid NumPy RandomState state") from exc
    return probe.get_state()


@contextmanager
def fork_rng(seed=None, *, state=None):
    """Temporarily fork NumPy's global RNG and restore the caller state exactly.

    With neither argument, the context starts from the caller's current RNG state, so
    draws inside it predict the caller's next draws without consuming them. ``seed``
    starts a deterministic stream, while ``state`` replays an exact state previously
    returned by ``np.random.get_state()``. Seed and state are mutually exclusive.

    State validation happens on an isolated ``RandomState`` before the process-global
    RNG is inspected or changed. Nested contexts are reentrant, and overlapping threads
    are serialized because NumPy's legacy RNG is process-global. Exceptions restore the
    state observed on entry.
    """
    seed = _validate_seed(seed)
    if seed is not None and state is not None:
        raise ValueError("seed and state are mutually exclusive")
    state = _validate_state(state)

    with _RNG_FORK_LOCK:
        caller_state = np.random.get_state()
        try:
            if state is not None:
                np.random.set_state(state)
            elif seed is not None:
                np.random.seed(seed)
            yield
        finally:
            np.random.set_state(caller_state)
