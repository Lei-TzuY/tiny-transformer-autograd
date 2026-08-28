"""Exact replay from independent MT19937 bit-generator state."""

import numpy as np

from engine import fork_rng


def _state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_fork_rng_replays_mt19937_source_without_consuming_it():
    source = np.random.MT19937(123)
    source.random_raw(7)
    source_before = source.state

    expected = np.random.RandomState()
    expected.set_state(source_before)
    expected_values = expected.random_sample(5)

    np.random.seed(991)
    caller_before = np.random.get_state()
    with fork_rng(state=source):
        actual_values = np.random.random_sample(5)
    caller_after = np.random.get_state()

    np.testing.assert_array_equal(actual_values, expected_values)
    assert source.state["bit_generator"] == source_before["bit_generator"]
    np.testing.assert_array_equal(
        source.state["state"]["key"], source_before["state"]["key"]
    )
    assert source.state["state"]["pos"] == source_before["state"]["pos"]
    assert _state_equal(caller_after, caller_before)


def test_fork_rng_snapshots_mt19937_source_before_context_body_mutation():
    source = np.random.MT19937(456)
    source.random_raw(3)
    snapshot = source.state

    expected = np.random.RandomState()
    expected.set_state(snapshot)
    expected_values = expected.random_sample(4)

    with fork_rng(state=source):
        source.random_raw(50)
        actual_values = np.random.random_sample(4)

    np.testing.assert_array_equal(actual_values, expected_values)


def test_non_mt19937_bit_generator_is_rejected_without_global_rng_mutation():
    source = np.random.PCG64(123)
    np.random.seed(812)
    before = np.random.get_state()

    try:
        with fork_rng(state=source):
            raise AssertionError("invalid source must not enter the context")
    except ValueError as exc:
        assert str(exc) == "state must be a valid NumPy RandomState state"
    else:
        raise AssertionError("expected invalid bit-generator rejection")

    after = np.random.get_state()
    assert _state_equal(after, before)
