"""Regression tests for RandomState-backed NumPy RNG forks."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import fork_rng


def _state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_randomstate_source_replays_exact_current_position_without_consuming_source():
    source = np.random.RandomState(123)
    source.random_sample(7)
    source_before = source.get_state()

    expected = np.random.RandomState()
    expected.set_state(source_before)
    expected_values = expected.random_sample(5)

    np.random.seed(999)
    caller_before = np.random.get_state()
    with fork_rng(state=source):
        actual_values = np.random.random_sample(5)

    np.testing.assert_array_equal(actual_values, expected_values)
    assert _state_equal(source.get_state(), source_before)
    assert _state_equal(np.random.get_state(), caller_before)


def test_randomstate_source_is_snapshotted_before_context_entry():
    source = np.random.RandomState(77)
    source.random_sample(3)
    source_state = source.get_state()

    expected = np.random.RandomState()
    expected.set_state(source_state)
    expected_values = expected.randint(0, 1000, size=6)

    with fork_rng(state=source):
        source.random_sample(20)
        actual_values = np.random.randint(0, 1000, size=6)

    np.testing.assert_array_equal(actual_values, expected_values)


def test_generator_object_remains_an_explicitly_invalid_state_source():
    generator = np.random.default_rng(123)
    np.random.seed(456)
    caller_before = np.random.get_state()

    try:
        with fork_rng(state=generator):
            raise AssertionError("invalid Generator state source entered the context")
    except ValueError as exc:
        assert str(exc) == "state must be a valid NumPy RandomState state"
    else:
        raise AssertionError("Generator state source should be rejected")

    assert _state_equal(np.random.get_state(), caller_before)
