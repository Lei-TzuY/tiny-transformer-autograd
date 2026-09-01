"""Concurrency regressions for KVCacheBuffer."""

import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.kv_cache import KVCacheBuffer


def _token(value):
    return np.array([[[float(value)]]], dtype=np.float64)


def test_concurrent_appends_are_serializable_and_never_lose_a_chunk():
    cache = KVCacheBuffer(4)
    barrier = threading.Barrier(3)
    errors = []

    def worker(value):
        try:
            barrier.wait(timeout=5)
            cache.append(_token(value), _token(value * 10))
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)

    left = threading.Thread(target=worker, args=(1,))
    right = threading.Thread(target=worker, args=(2,))
    left.start()
    right.start()
    barrier.wait(timeout=5)
    left.join(timeout=5)
    right.join(timeout=5)

    assert not left.is_alive()
    assert not right.is_alive()
    assert errors == []
    assert cache.length == 2
    snapshot = cache.snapshot()
    key = snapshot["k"].reshape(-1).tolist()
    value = snapshot["v"].reshape(-1).tolist()
    assert (key, value) in [
        ([1.0, 2.0], [10.0, 20.0]),
        ([2.0, 1.0], [20.0, 10.0]),
    ]


def test_snapshot_reader_never_observes_key_value_partial_commit():
    cache = KVCacheBuffer(64)
    cache.append(_token(0), _token(0))
    start = threading.Barrier(2)
    stop = threading.Event()
    errors = []
    observations = []

    def writer():
        try:
            start.wait(timeout=5)
            for index in range(1, 40):
                cache.append(_token(index), _token(index * 10))
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    thread = threading.Thread(target=writer)
    thread.start()
    start.wait(timeout=5)

    while not stop.is_set():
        snap = cache.snapshot()
        observations.append(snap)
    thread.join(timeout=5)
    observations.append(cache.snapshot())

    assert not thread.is_alive()
    assert errors == []
    assert observations
    for snap in observations:
        assert snap["k"].shape[-2] == snap["v"].shape[-2]
        np.testing.assert_array_equal(snap["v"], snap["k"] * 10.0)


def test_concurrent_capacity_race_has_one_commit_and_one_clean_failure():
    cache = KVCacheBuffer(1)
    barrier = threading.Barrier(3)
    successes = []
    failures = []

    def worker(value):
        barrier.wait(timeout=5)
        try:
            cache.append(_token(value), _token(value + 100))
            successes.append(value)
        except OverflowError:
            failures.append(value)

    one = threading.Thread(target=worker, args=(1,))
    two = threading.Thread(target=worker, args=(2,))
    one.start()
    two.start()
    barrier.wait(timeout=5)
    one.join(timeout=5)
    two.join(timeout=5)

    assert sorted(successes + failures) == [1, 2]
    assert len(successes) == 1
    assert len(failures) == 1
    assert cache.length == 1
    winning = successes[0]
    snap = cache.snapshot()
    assert snap["k"].item() == winning
    assert snap["v"].item() == winning + 100


def test_many_single_token_appends_keep_backing_identity_fixed():
    cache = KVCacheBuffer(32)
    cache.append(_token(0), _token(0))
    k_storage = cache._k_storage
    v_storage = cache._v_storage
    k_ptr = cache._k_storage.__array_interface__["data"][0]
    v_ptr = cache._v_storage.__array_interface__["data"][0]

    for value in range(1, 32):
        cache.append(_token(value), _token(-value))
        assert cache._k_storage is k_storage
        assert cache._v_storage is v_storage
        assert cache._k_storage.__array_interface__["data"][0] == k_ptr
        assert cache._v_storage.__array_interface__["data"][0] == v_ptr

    assert cache.length == 32
    np.testing.assert_array_equal(cache.snapshot()["k"].reshape(-1), np.arange(32.0))


def test_reentrant_same_thread_operations_do_not_deadlock():
    cache = KVCacheBuffer(4)
    with cache._lock:
        cache.append(_token(1), _token(2))
        assert cache.length == 1
        snap = cache.snapshot()
        assert snap["k"].item() == 1.0
        cache.truncate(0)
        cache.append(_token(3), _token(4))
    assert cache.snapshot()["k"].item() == 3.0
