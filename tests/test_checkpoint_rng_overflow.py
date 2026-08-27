"""Normalize overflow failures while validating checkpoint NumPy RNG state."""

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


def _overflowing_rng_state(field):
    state = list(np.random.RandomState(0).get_state())
    index = {"pos": 2, "has_gauss": 3, "cached_gaussian": 4}[field]
    state[index] = 10**400
    return tuple(state)


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("field", ["pos", "has_gauss", "cached_gaussian"])
def test_restore_normalizes_rng_overflow_before_snapshot_and_without_rng_mutation(field):
    np.random.seed(2026)
    before = np.random.get_state()
    state = {
        "format_version": 2,
        "model": {},
        "rng_state": _overflowing_rng_state(field),
    }

    with pytest.raises(ValueError, match="invalid checkpoint NumPy RNG state"):
        restore_checkpoint(state, _SnapshotMustNotRun())

    _assert_rng_equal(np.random.get_state(), before)


def test_read_normalizes_rng_overflow(tmp_path):
    destination = tmp_path / "overflow-rng.ckpt"
    state = {
        "format_version": 2,
        "model": {},
        "rng_state": _overflowing_rng_state("pos"),
    }
    with destination.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="invalid checkpoint NumPy RNG state"):
        read_checkpoint(destination)


def test_valid_rng_state_still_round_trips_through_restore():
    source = np.random.RandomState(123).get_state()
    state = {
        "format_version": 2,
        "model": {},
        "rng_state": source,
    }

    class _EmptyModel:
        def state_dict(self):
            return {}

        def load_state_dict(self, incoming, strict=True):
            assert incoming == {}
            assert strict is True

    restore_checkpoint(state, _EmptyModel())
    restored = np.random.get_state()
    _assert_rng_equal(restored, source)
