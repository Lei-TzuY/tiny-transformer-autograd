"""Regression tests for exception-safe NumPy RNG isolation."""

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.rng import fork_rng


def _assert_rng_state_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_seeded_fork_is_deterministic_and_restores_caller_state():
    np.random.seed(11)
    before = np.random.get_state()

    with fork_rng(7):
        actual = np.random.random(5)

    _assert_rng_state_equal(np.random.get_state(), before)

    np.random.seed(7)
    expected = np.random.random(5)
    np.testing.assert_array_equal(actual, expected)


def test_unseeded_fork_predicts_next_draws_without_consuming_them():
    np.random.seed(23)

    with fork_rng():
        inside = np.random.randint(0, 1000, size=8)

    outside = np.random.randint(0, 1000, size=8)
    np.testing.assert_array_equal(inside, outside)


def test_nested_forks_restore_the_outer_stream_exactly():
    with fork_rng(101):
        first = np.random.random(3)
        outer_state = np.random.get_state()

        with fork_rng(202):
            nested = np.random.random(3)

        _assert_rng_state_equal(np.random.get_state(), outer_state)
        second = np.random.random(3)

    np.random.seed(101)
    np.testing.assert_array_equal(first, np.random.random(3))
    np.testing.assert_array_equal(second, np.random.random(3))

    np.random.seed(202)
    np.testing.assert_array_equal(nested, np.random.random(3))


def test_overlapping_threaded_forks_are_serialized_and_restore_caller_state():
    np.random.seed(29)
    before = np.random.get_state()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    values = {}

    def first_worker():
        with fork_rng(101):
            values["first"] = np.random.random(4)
            first_entered.set()
            assert release_first.wait(timeout=5.0)

    def second_worker():
        assert first_entered.wait(timeout=5.0)
        second_attempting.set()
        with fork_rng(202):
            second_entered.set()
            values["second"] = np.random.random(4)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()

    assert first_entered.wait(timeout=5.0)
    assert second_attempting.wait(timeout=5.0)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    _assert_rng_state_equal(np.random.get_state(), before)

    np.random.seed(101)
    np.testing.assert_array_equal(values["first"], np.random.random(4))
    np.random.seed(202)
    np.testing.assert_array_equal(values["second"], np.random.random(4))


def test_exception_restores_rng_state():
    np.random.seed(37)
    before = np.random.get_state()

    with pytest.raises(RuntimeError, match="boom"):
        with fork_rng(5):
            np.random.normal(size=4)
            raise RuntimeError("boom")

    _assert_rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize("seed", [0, 2**32 - 1, np.int64(17), np.uint32(19)])
def test_accepts_supported_integer_seeds(seed):
    with fork_rng(seed):
        values = np.random.random(2)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.5, "7", object()])
def test_rejects_non_integer_seeds_without_changing_rng(seed):
    np.random.seed(41)
    before = np.random.get_state()

    with pytest.raises(TypeError, match="seed must be an integer or None"):
        with fork_rng(seed):
            pytest.fail("invalid seed entered the context")

    _assert_rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize("seed", [-1, 2**32, -(2**80), 2**80])
def test_rejects_out_of_range_seeds_without_changing_rng(seed):
    np.random.seed(43)
    before = np.random.get_state()

    with pytest.raises(ValueError, match=r"seed must be in \[0, 4294967295\]"):
        with fork_rng(seed):
            pytest.fail("invalid seed entered the context")

    _assert_rng_state_equal(np.random.get_state(), before)
