"""Regression coverage for exact NumPy RandomState replay through ``fork_rng``."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import fork_rng


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_explicit_state_replays_exact_stream_and_restores_caller():
    np.random.seed(123)
    np.random.normal(size=5)
    replay_state = np.random.get_state()
    expected = np.random.normal(size=8)

    np.random.seed(999)
    caller_state = np.random.get_state()
    with fork_rng(state=replay_state):
        actual = np.random.normal(size=8)

    np.testing.assert_array_equal(actual, expected)
    _assert_rng_state_equal(np.random.get_state(), caller_state)


def test_explicit_state_is_copied_during_validation():
    np.random.seed(321)
    replay_state = np.random.get_state()
    expected_probe = np.random.RandomState()
    expected_probe.set_state(replay_state)
    expected = expected_probe.randint(0, 1000, size=6)

    key = replay_state[1]
    with fork_rng(state=replay_state):
        key[:] = 0
        actual = np.random.randint(0, 1000, size=6)

    np.testing.assert_array_equal(actual, expected)


def test_invalid_state_fails_without_touching_global_rng():
    np.random.seed(77)
    before = np.random.get_state()

    with pytest.raises(ValueError, match="state must be a valid NumPy RandomState state"):
        with fork_rng(state=("MT19937", np.array([1, 2, 3]), 0, 0, 0.0)):
            raise AssertionError("invalid state must fail before entering the context")

    _assert_rng_state_equal(np.random.get_state(), before)


def test_seed_and_state_are_mutually_exclusive_before_global_rng_access():
    np.random.seed(88)
    replay_state = np.random.get_state()
    before = np.random.get_state()

    with pytest.raises(ValueError, match="seed and state are mutually exclusive"):
        with fork_rng(1, state=replay_state):
            raise AssertionError("conflicting replay sources must not enter the context")

    _assert_rng_state_equal(np.random.get_state(), before)


def test_explicit_state_restores_caller_after_exception():
    np.random.seed(55)
    replay_state = np.random.get_state()

    np.random.seed(66)
    caller_state = np.random.get_state()
    with pytest.raises(RuntimeError, match="boom"):
        with fork_rng(state=replay_state):
            np.random.random_sample(4)
            raise RuntimeError("boom")

    _assert_rng_state_equal(np.random.get_state(), caller_state)
