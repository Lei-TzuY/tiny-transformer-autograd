"""Validation and fail-closed behavior for KVCacheBuffer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.kv_cache import KVCacheBuffer


def _chunk(shape=(1, 1, 3), dtype=np.float64, value=1.0):
    return np.full(shape, value, dtype=dtype)


def _snapshot_state(cache):
    return (
        cache.length,
        cache.remaining,
        cache.storage_nbytes,
        None if cache._k_storage is None else id(cache._k_storage),
        None if cache._v_storage is None else id(cache._v_storage),
        None if cache._k_storage is None else cache.snapshot(),
    )


def _assert_state_equal(cache, before):
    length, remaining, storage_nbytes, k_id, v_id, snapshot = before
    assert cache.length == length
    assert cache.remaining == remaining
    assert cache.storage_nbytes == storage_nbytes
    assert (None if cache._k_storage is None else id(cache._k_storage)) == k_id
    assert (None if cache._v_storage is None else id(cache._v_storage)) == v_id
    if snapshot is not None:
        current = cache.snapshot()
        np.testing.assert_array_equal(current["k"], snapshot["k"])
        np.testing.assert_array_equal(current["v"], snapshot["v"])


def test_capacity_validation_is_strict():
    for value in (True, np.bool_(False), 1.5, "3", None):
        with pytest.raises(TypeError, match="positive integer"):
            KVCacheBuffer(value)
    for value in (0, -1, np.int64(-3)):
        with pytest.raises(ValueError, match="positive"):
            KVCacheBuffer(value)


def test_view_and_snapshot_require_initialized_layout():
    cache = KVCacheBuffer(3)
    with pytest.raises(RuntimeError, match="not initialized"):
        cache.view()
    with pytest.raises(RuntimeError, match="not initialized"):
        cache.snapshot()


def test_truncate_validation_before_and_after_initialization():
    cache = KVCacheBuffer(3)
    for value in (True, np.bool_(False), 1.5, "1"):
        with pytest.raises(TypeError, match="non-negative integer"):
            cache.truncate(value)
    with pytest.raises(ValueError, match="non-negative"):
        cache.truncate(-1)
    assert cache.truncate(0) is cache
    with pytest.raises(ValueError, match="uninitialized"):
        cache.truncate(1)

    cache.append(_chunk(shape=(1, 2, 3)), _chunk(shape=(1, 2, 3), value=2.0))
    with pytest.raises(ValueError, match="cannot extend"):
        cache.truncate(3)
    assert cache.length == 2


def test_append_requires_numpy_floating_arrays():
    cache = KVCacheBuffer(3)
    invalid = [
        ([[[1.0]]], _chunk()),
        (_chunk(), [[[1.0]]]),
        (np.ones((1, 1, 1), dtype=np.int64), _chunk()),
        (_chunk(), np.ones((1, 1, 1), dtype=np.int64)),
        (np.ones((1, 1, 1), dtype=bool), _chunk()),
        (_chunk(), np.ones((1, 1, 1), dtype=complex)),
    ]
    for key, value in invalid:
        with pytest.raises(TypeError):
            cache.append(key, value)
        assert cache.initialized is False


def test_append_requires_rank_two_or_more_and_nonempty_time():
    cache = KVCacheBuffer(3)
    with pytest.raises(ValueError, match="at least two"):
        cache.append(np.ones((3,), dtype=float), np.ones((3,), dtype=float))
    with pytest.raises(ValueError, match="at least one time step"):
        cache.append(np.empty((1, 0, 3)), np.empty((1, 0, 3)))
    assert cache.initialized is False


def test_append_rejects_nonfinite_sources_transactionally():
    cache = KVCacheBuffer(4)
    cache.append(_chunk(shape=(1, 2, 2)), _chunk(shape=(1, 2, 2), value=4.0))
    before = _snapshot_state(cache)

    for bad in (np.nan, np.inf, -np.inf):
        key = _chunk(shape=(1, 1, 2))
        key[0, 0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            cache.append(key, _chunk(shape=(1, 1, 2)))
        _assert_state_equal(cache, before)


def test_key_value_rank_leading_dimensions_and_time_must_match():
    cases = [
        (_chunk(shape=(1, 1, 2)), _chunk(shape=(1, 1, 1, 2)), "same rank"),
        (_chunk(shape=(1, 1, 2)), _chunk(shape=(2, 1, 2)), "leading"),
        (_chunk(shape=(1, 1, 2)), _chunk(shape=(1, 2, 5)), "same number"),
    ]
    for key, value, message in cases:
        cache = KVCacheBuffer(4)
        with pytest.raises(ValueError, match=message):
            cache.append(key, value)
        assert cache.initialized is False


def test_initialized_layout_and_dtype_drift_fail_before_write():
    cache = KVCacheBuffer(5)
    cache.append(
        _chunk(shape=(2, 3, 1, 4), dtype=np.float32),
        _chunk(shape=(2, 3, 1, 6), dtype=np.float64),
    )
    before = _snapshot_state(cache)

    bad_cases = [
        (
            _chunk(shape=(1, 3, 1, 4), dtype=np.float32),
            _chunk(shape=(1, 3, 1, 6), dtype=np.float64),
            ValueError,
        ),
        (
            _chunk(shape=(2, 2, 1, 4), dtype=np.float32),
            _chunk(shape=(2, 2, 1, 6), dtype=np.float64),
            ValueError,
        ),
        (
            _chunk(shape=(2, 3, 1, 5), dtype=np.float32),
            _chunk(shape=(2, 3, 1, 6), dtype=np.float64),
            ValueError,
        ),
        (
            _chunk(shape=(2, 3, 1, 4), dtype=np.float64),
            _chunk(shape=(2, 3, 1, 6), dtype=np.float64),
            TypeError,
        ),
        (
            _chunk(shape=(2, 3, 1, 4), dtype=np.float32),
            _chunk(shape=(2, 3, 1, 6), dtype=np.float32),
            TypeError,
        ),
    ]
    for key, value, error in bad_cases:
        with pytest.raises(error):
            cache.append(key, value)
        _assert_state_equal(cache, before)


def test_capacity_overflow_is_transactional():
    cache = KVCacheBuffer(3)
    cache.append(_chunk(shape=(1, 2, 2)), _chunk(shape=(1, 2, 2), value=3.0))
    before = _snapshot_state(cache)

    with pytest.raises(OverflowError, match="capacity 3"):
        cache.append(_chunk(shape=(1, 2, 2), value=5.0), _chunk(shape=(1, 2, 2), value=6.0))
    _assert_state_equal(cache, before)


def test_first_append_capacity_overflow_does_not_initialize_storage():
    cache = KVCacheBuffer(1)
    with pytest.raises(OverflowError):
        cache.append(_chunk(shape=(1, 2, 3)), _chunk(shape=(1, 2, 3)))
    assert cache.initialized is False
    assert cache.storage_nbytes == 0
    assert cache.length == 0


def test_source_validation_precedes_capacity_overflow():
    cache = KVCacheBuffer(1)
    key = _chunk(shape=(1, 2, 3))
    key[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cache.append(key, _chunk(shape=(1, 2, 3)))
    assert cache.initialized is False


def test_corrupted_internal_readonly_state_fails_before_either_write():
    cache = KVCacheBuffer(4)
    cache.append(_chunk(shape=(1, 1, 2)), _chunk(shape=(1, 1, 2), value=2.0))
    before = cache.snapshot()
    cache._v_storage.flags.writeable = False

    with pytest.raises(RuntimeError, match="read-only"):
        cache.append(_chunk(shape=(1, 1, 2), value=7.0), _chunk(shape=(1, 1, 2), value=8.0))

    assert cache.length == 1
    np.testing.assert_array_equal(cache._k_storage[..., :1, :], before["k"])
    np.testing.assert_array_equal(cache._v_storage[..., :1, :], before["v"])


def test_corrupted_internal_metadata_fails_closed():
    cache = KVCacheBuffer(4)
    cache.append(_chunk(), _chunk(value=2.0))
    cache._length = 99
    with pytest.raises(RuntimeError, match="outside capacity"):
        cache.view()


def test_global_numpy_rng_is_untouched_by_all_buffer_operations():
    np.random.seed(918273)
    before = np.random.get_state()

    cache = KVCacheBuffer(4)
    cache.append(_chunk(shape=(1, 2, 3)), _chunk(shape=(1, 2, 3), value=2.0))
    cache.view()
    cache.snapshot()
    cache.truncate(1)
    cache.clear()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
