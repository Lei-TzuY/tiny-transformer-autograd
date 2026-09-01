import numpy as np

from nn import (
    KVCacheBuffer,
    MultiHeadAttention,
    RotaryEmbedding,
    SelfAttention,
    infer_with_kv_buffer,
)


def _ptr(array):
    return int(array.__array_interface__["data"][0])


def test_self_attention_buffer_matches_legacy_cached_decode():
    np.random.seed(1201)
    attention = SelfAttention(4)
    prefix = np.random.randn(2, 3, 4)
    token = np.random.randn(2, 1, 4)

    legacy_prefix, legacy_cache = attention.infer(prefix)
    legacy_token, legacy_cache = attention.infer(token, cache=legacy_cache)

    buffer = KVCacheBuffer(8)
    buffered_prefix, live = infer_with_kv_buffer(attention, prefix, buffer)
    k_ptr = _ptr(buffer._k_storage)
    v_ptr = _ptr(buffer._v_storage)
    buffered_token, live = infer_with_kv_buffer(attention, token, buffer)

    np.testing.assert_allclose(buffered_prefix, attention.infer(prefix)[0])
    np.testing.assert_allclose(buffered_token, legacy_token)
    np.testing.assert_allclose(live["k"], legacy_cache["k"])
    np.testing.assert_allclose(live["v"], legacy_cache["v"])
    assert buffer.length == 4
    assert _ptr(buffer._k_storage) == k_ptr
    assert _ptr(buffer._v_storage) == v_ptr
    assert not live["k"].flags.writeable
    assert not live["v"].flags.writeable


def test_multi_head_buffer_matches_legacy_across_multiple_chunks():
    np.random.seed(1202)
    attention = MultiHeadAttention(8, 2)
    chunks = [
        np.random.randn(2, 2, 8),
        np.random.randn(2, 1, 8),
        np.random.randn(2, 2, 8),
    ]

    legacy_cache = None
    legacy_outputs = []
    for chunk in chunks:
        output, legacy_cache = attention.infer(chunk, cache=legacy_cache)
        legacy_outputs.append(output)

    buffer = KVCacheBuffer(8)
    buffered_outputs = []
    pointers = None
    for chunk in chunks:
        output, live = infer_with_kv_buffer(attention, chunk, buffer)
        buffered_outputs.append(output)
        current = (_ptr(buffer._k_storage), _ptr(buffer._v_storage))
        if pointers is None:
            pointers = current
        assert current == pointers

    for actual, expected in zip(buffered_outputs, legacy_outputs):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(live["k"], legacy_cache["k"])
    np.testing.assert_allclose(live["v"], legacy_cache["v"])
    assert buffer.length == 5


def test_rope_multi_head_buffer_matches_legacy_with_explicit_positions():
    np.random.seed(1203)
    rope = RotaryEmbedding(dim=4, max_pos=16)
    attention = MultiHeadAttention(8, 2, rope=rope)
    prefix = np.random.randn(2, 2, 8)
    token = np.random.randn(2, 1, 8)
    prefix_positions = np.array([[[0, 1]], [[2, 3]]], dtype=np.int64)
    token_positions = np.array([[[2]], [[4]]], dtype=np.int64)

    legacy_prefix, legacy_cache = attention.infer(
        prefix,
        positions=prefix_positions,
    )
    legacy_token, legacy_cache = attention.infer(
        token,
        cache=legacy_cache,
        positions=token_positions,
    )

    buffer = KVCacheBuffer(8)
    buffered_prefix, _ = infer_with_kv_buffer(
        attention,
        prefix,
        buffer,
        positions=prefix_positions,
    )
    buffered_token, live = infer_with_kv_buffer(
        attention,
        token,
        buffer,
        positions=token_positions,
    )

    np.testing.assert_allclose(buffered_prefix, legacy_prefix)
    np.testing.assert_allclose(buffered_token, legacy_token)
    np.testing.assert_allclose(live["k"], legacy_cache["k"])
    np.testing.assert_allclose(live["v"], legacy_cache["v"])


def test_key_bias_semantics_match_legacy_cache():
    np.random.seed(1204)
    attention = MultiHeadAttention(8, 2)
    prefix = np.random.randn(1, 2, 8)
    token = np.random.randn(1, 1, 8)

    _, legacy_cache = attention.infer(prefix)
    key_bias = np.array([[[0.0, -np.inf, 0.0]]])
    expected, legacy_cache = attention.infer(
        token,
        cache=legacy_cache,
        key_bias=key_bias,
    )

    buffer = KVCacheBuffer(4)
    infer_with_kv_buffer(attention, prefix, buffer)
    actual, live = infer_with_kv_buffer(
        attention,
        token,
        buffer,
        key_bias=key_bias,
    )

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(live["k"], legacy_cache["k"])
    np.testing.assert_allclose(live["v"], legacy_cache["v"])


def test_clear_reuses_storage_and_restarts_rope_offset():
    np.random.seed(1205)
    rope = RotaryEmbedding(dim=4, max_pos=16)
    attention = MultiHeadAttention(8, 2, rope=rope)
    chunk = np.random.randn(1, 2, 8)

    first, _ = infer_with_kv_buffer(attention, chunk, KVCacheBuffer(4))

    buffer = KVCacheBuffer(4)
    infer_with_kv_buffer(attention, np.random.randn(1, 1, 8), buffer)
    pointers = (_ptr(buffer._k_storage), _ptr(buffer._v_storage))
    buffer.clear()
    replay, live = infer_with_kv_buffer(attention, chunk, buffer)

    np.testing.assert_allclose(replay, first)
    assert buffer.length == 2
    assert (_ptr(buffer._k_storage), _ptr(buffer._v_storage)) == pointers
    assert live["k"].shape == (1, 2, 2, 4)
