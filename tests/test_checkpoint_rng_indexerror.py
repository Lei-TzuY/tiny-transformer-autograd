"""Normalize NumPy-specific failures while validating checkpoint RNG state."""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, restore_checkpoint


class _SnapshotMustNotRun:
    def state_dict(self):
        raise AssertionError("model snapshot must not run for an invalid envelope")


def _short_mt19937_state():
    return ("MT19937", np.array([1], dtype=np.uint32), 0, 0, 0.0)


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_restore_normalizes_short_mt19937_indexerror_without_global_rng_mutation():
    np.random.seed(2026)
    before = np.random.get_state()
    state = {
        "format_version": 2,
        "model": {},
        "rng_state": _short_mt19937_state(),
    }

    with pytest.raises(ValueError, match="invalid checkpoint NumPy RNG state"):
        restore_checkpoint(state, _SnapshotMustNotRun())

    _assert_rng_equal(np.random.get_state(), before)


def test_read_normalizes_short_mt19937_indexerror(tmp_path):
    destination = tmp_path / "short-rng.ckpt"
    state = {
        "format_version": 2,
        "model": {},
        "rng_state": _short_mt19937_state(),
    }
    with destination.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="invalid checkpoint NumPy RNG state"):
        read_checkpoint(destination)
