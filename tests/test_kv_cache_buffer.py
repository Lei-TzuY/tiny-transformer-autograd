"""Core behavior for the fixed-capacity KV cache buffer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.kv_cache import KVCacheBuffer


def test_single_head_chunks_append_without_reallocating_history_storage():
    cache = KVCacheBuffer(6)
    first_k = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    first_v = (100 + np.arange(6, dtype=np.float64)).reshape(1, 2, 3)

    first = cache.append(first_k, first_v)
    k_storage = cache._k_storage
    v_storage = cache._v_storage
    assert first["k"].shape == (1, 2, 3)
    assert first["v"].shape == (1, 2, 3)
    np.testing.assert_array_equal(first["k"], first_k)
    np.testing.assert_array_equal(first["v"], first_v)

    second_k = np.full((1, 1, 3), 7.0)
    second_v = np.full((1, 1, 3), 8.0)
    second = cache.append(second_k, second_v)

    assert cache._k_storage is k_storage
    assert cache._v_storage is v_storage
    assert cache.length == 3
    assert cache.remaining == 3
    np.testing.assert_array_equal(second["k"], np.concatenate([first_k, second_k], axis=-2))
    np.testing.assert_array_equal(second["v"], np.concatenate([first_v, second_v], axis=-2))


def test_multi_head_layout_uses_penultimate_time_axis():
    cache = KVCacheBuffer(5)
    key = np.arange(2 * 3 * 2 * 4, dtype=np.float64).reshape(2, 3, 2, 4)
    value = key + 1000.0

    view = cache.append(key, value)
    assert view["k"].shape == (2, 3, 2, 4)
    assert cache._k_storage.shape == (2, 3, 5, 4)
    assert cache._v_storage.shape == (2, 3, 5, 4)


def test_key_and_value_feature_widths_may_differ():
    cache = KVCacheBuffer(4)
    key = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    value = np.arange(20, dtype=np.float64).reshape(2, 2, 5)

    view = cache.append(key, value)
    assert view["k"].shape == (2, 2, 3)
    assert view["v"].shape == (2, 2, 5)
    assert view["k"].dtype == np.float32
    assert view["v"].dtype == np.float64
    assert cache._k_storage.shape == (2, 4, 3)
    assert cache._v_storage.shape == (2, 4, 5)


def test_view_is_read_only_and_snapshot_is_writable_independent_storage():
    cache = KVCacheBuffer(4)
    cache.append(np.ones((1, 2, 3)), np.full((1, 2, 3), 2.0))

    view = cache.view()
    assert not view["k"].flags.writeable
    assert not view["v"].flags.writeable
    with pytest.raises(ValueError):
        view["k"][0, 0, 0] = 9.0

    snapshot = cache.snapshot()
    assert snapshot["k"].flags.writeable
    assert snapshot["v"].flags.writeable
    assert not np.shares_memory(snapshot["k"], cache._k_storage)
    assert not np.shares_memory(snapshot["v"], cache._v_storage)
    snapshot["k"][...] = 99.0
    assert np.all(cache.view()["k"] == 1.0)


def test_retained_live_view_shares_storage_and_snapshot_survives_reuse():
    cache = KVCacheBuffer(3)
    cache.append(np.array([[[1.0]]]), np.array([[[10.0]]]))
    live = cache.view()
    stable = cache.snapshot()

    cache.clear()
    cache.append(np.array([[[2.0]]]), np.array([[[20.0]]]))

    # A live view intentionally aliases storage; callers needing retained ownership
    # must use snapshot().
    assert live["k"].item() == 2.0
    assert live["v"].item() == 20.0
    assert stable["k"].item() == 1.0
    assert stable["v"].item() == 10.0


def test_truncate_and_reappend_reuses_capacity_without_reallocation():
    cache = KVCacheBuffer(5)
    key = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
    value = key + 20.0
    cache.append(key, value)
    k_storage = cache._k_storage
    v_storage = cache._v_storage

    assert cache.truncate(2) is cache
    np.testing.assert_array_equal(cache.view()["k"], key[:, :2])
    cache.append(np.full((1, 2, 3), -1.0), np.full((1, 2, 3), -2.0))

    assert cache._k_storage is k_storage
    assert cache._v_storage is v_storage
    expected_k = np.concatenate([key[:, :2], np.full((1, 2, 3), -1.0)], axis=1)
    expected_v = np.concatenate([value[:, :2], np.full((1, 2, 3), -2.0)], axis=1)
    np.testing.assert_array_equal(cache.view()["k"], expected_k)
    np.testing.assert_array_equal(cache.view()["v"], expected_v)


def test_clear_retains_allocation_and_layout_for_reuse():
    cache = KVCacheBuffer(3)
    cache.append(np.ones((1, 2, 4), dtype=np.float32), np.ones((1, 2, 6), dtype=np.float64))
    k_storage = cache._k_storage
    v_storage = cache._v_storage
    reserved = cache.storage_nbytes

    assert cache.clear() is cache
    assert cache.length == 0
    assert cache.remaining == 3
    assert cache.storage_nbytes == reserved
    assert cache.live_nbytes == 0
    assert cache.view()["k"].shape == (1, 0, 4)
    assert cache.view()["v"].shape == (1, 0, 6)

    cache.append(np.zeros((1, 1, 4), dtype=np.float32), np.zeros((1, 1, 6), dtype=np.float64))
    assert cache._k_storage is k_storage
    assert cache._v_storage is v_storage


def test_storage_and_live_byte_accounting_tracks_capacity_and_length():
    cache = KVCacheBuffer(7)
    assert cache.storage_nbytes == 0
    assert cache.live_nbytes == 0

    key = np.zeros((2, 3, 2, 4), dtype=np.float32)
    value = np.zeros((2, 3, 2, 5), dtype=np.float64)
    cache.append(key, value)

    expected_storage = (2 * 3 * 7 * 4 * 4) + (2 * 3 * 7 * 5 * 8)
    expected_live = (2 * 3 * 2 * 4 * 4) + (2 * 3 * 2 * 5 * 8)
    assert cache.storage_nbytes == expected_storage
    assert cache.live_nbytes == expected_live

    cache.truncate(1)
    assert cache.live_nbytes == expected_live // 2
    assert cache.storage_nbytes == expected_storage


def test_self_append_from_live_view_is_overlap_safe():
    cache = KVCacheBuffer(4)
    cache.append(
        np.array([[[1.0], [2.0]]]),
        np.array([[[10.0], [20.0]]]),
    )
    live = cache.view()

    result = cache.append(live["k"], live["v"])
    np.testing.assert_array_equal(result["k"], [[[1.0], [2.0], [1.0], [2.0]]])
    np.testing.assert_array_equal(result["v"], [[[10.0], [20.0], [10.0], [20.0]]])


def test_source_arrays_are_never_aliased_or_mutated():
    cache = KVCacheBuffer(3)
    key = np.array([[[1.0], [2.0]]])
    value = np.array([[[3.0], [4.0]]])
    key_before = key.copy()
    value_before = value.copy()

    cache.append(key, value)
    key[...] = 99.0
    value[...] = 88.0

    np.testing.assert_array_equal(cache.view()["k"], key_before)
    np.testing.assert_array_equal(cache.view()["v"], value_before)


def test_repr_len_and_properties_are_stable_before_and_after_initialization():
    cache = KVCacheBuffer(np.int64(3))
    assert len(cache) == 0
    assert cache.capacity == 3
    assert cache.remaining == 3
    assert cache.initialized is False
    assert repr(cache) == "KVCacheBuffer(capacity=3, length=0, uninitialized)"

    cache.append(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
    assert len(cache) == 1
    assert cache.initialized is True
    assert repr(cache) == "KVCacheBuffer(capacity=3, length=1, initialized)"
