"""Exception-safe isolation for the process-global NumPy random state."""

from contextlib import contextmanager

import numpy as np


_MAX_LEGACY_SEED = 2**32 - 1


def _validate_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    seed = int(seed)
    if seed < 0 or seed > _MAX_LEGACY_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_LEGACY_SEED}]")
    return seed


@contextmanager
def fork_rng(seed=None):
    """Temporarily fork NumPy's global RNG and restore the caller state exactly.

    With ``seed=None``, the context starts from the caller's current RNG state, so
    draws inside it predict the caller's next draws without consuming them. Supplying
    a seed instead starts an isolated deterministic stream. Nested contexts and
    exceptions are safe because each invocation restores the state it observed on
    entry.
    """
    seed = _validate_seed(seed)
    state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)
