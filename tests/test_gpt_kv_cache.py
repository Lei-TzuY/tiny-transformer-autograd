import numpy as np

from nn import GPT, GPTKVCache, infer_gpt_with_kv_cache


def _model(*, rope=False, layers=2):
    np.random.seed(1201)
    return GPT(
        vocab_size=23,
        context_len=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=layers,
        dropout=0.0,
        norm="rmsnorm" if rope else "layernorm",
        pos_encoding="rope" if rope else "learned",
        ffn="swiglu" if rope else "gelu",
    )


def _assert_cache_equal(buffered, legacy):
    snapshot = buffered.snapshot()
    assert snapshot is not None
    assert len(snapshot) == len(legacy)
    for actual, expected in zip(snapshot, legacy):
        np.testing.assert_array_equal(actual["k"], expected["k"])
        np.testing.assert_array_equal(actual["v"], expected["v"])
        assert not np.shares_memory(actual["k"], expected["k"])
        assert not np.shares_memory(actual["v"], expected["v"])


def test_buffered_gpt_prompt_and_incremental_decode_match_legacy_cache():
    model = _model()
    prompt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    next_one = np.array([[7], [8]], dtype=np.int64)
    next_two = np.array([[9], [10]], dtype=np.int64)

    legacy_logits, legacy_cache = model.infer(prompt)
    cache = GPTKVCache(model)
    buffered_logits, returned = infer_gpt_with_kv_cache(model, prompt, cache)
    assert returned is cache
    np.testing.assert_allclose(buffered_logits, legacy_logits, rtol=0.0, atol=0.0)
    _assert_cache_equal(cache, legacy_cache)

    pointers = [
        (buffer._k_storage.__array_interface__["data"][0], buffer._v_storage.__array_interface__["data"][0])
        for buffer in cache._buffers
    ]

    for token in (next_one, next_two):
        legacy_logits, legacy_cache = model.infer(token, legacy_cache)
        buffered_logits, _ = infer_gpt_with_kv_cache(model, token, cache)
        np.testing.assert_allclose(buffered_logits, legacy_logits, rtol=0.0, atol=0.0)
        _assert_cache_equal(cache, legacy_cache)

    assert cache.length == 5
    assert [
        (buffer._k_storage.__array_interface__["data"][0], buffer._v_storage.__array_interface__["data"][0])
        for buffer in cache._buffers
    ] == pointers


def test_rope_left_padded_buffered_inference_matches_legacy():
    model = _model(rope=True)
    prompt = np.array([[0, 2, 3], [4, 5, 6]], dtype=np.int64)
    keep = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)
    positions = np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64)

    legacy_logits, legacy_cache = model.infer(
        prompt,
        attention_mask=keep,
        position_ids=positions,
    )
    cache = GPTKVCache(model)
    buffered_logits, _ = infer_gpt_with_kv_cache(
        model,
        prompt,
        cache,
        attention_mask=keep,
        position_ids=positions,
    )
    np.testing.assert_allclose(buffered_logits, legacy_logits, rtol=0.0, atol=0.0)
    _assert_cache_equal(cache, legacy_cache)

    token = np.array([[7], [8]], dtype=np.int64)
    next_keep = np.concatenate([keep, np.ones((2, 1), dtype=bool)], axis=1)
    next_positions = np.array([[2], [3]], dtype=np.int64)
    legacy_logits, legacy_cache = model.infer(
        token,
        legacy_cache,
        attention_mask=next_keep,
        position_ids=next_positions,
    )
    buffered_logits, _ = infer_gpt_with_kv_cache(
        model,
        token,
        cache,
        attention_mask=next_keep,
        position_ids=next_positions,
    )
    np.testing.assert_allclose(buffered_logits, legacy_logits, rtol=0.0, atol=0.0)
    _assert_cache_equal(cache, legacy_cache)


def test_clear_reuses_all_layer_storage_and_restarts_positions():
    model = _model(rope=True, layers=3)
    cache = GPTKVCache(model)
    prompt = np.array([[1, 2, 3]], dtype=np.int64)

    first, _ = infer_gpt_with_kv_cache(model, prompt, cache)
    pointers = [
        (id(buffer._k_storage), id(buffer._v_storage)) for buffer in cache._buffers
    ]
    reserved = cache.storage_nbytes

    cache.clear()
    assert cache.initialized is True
    assert cache.length == 0
    assert cache.live_nbytes == 0
    assert cache.storage_nbytes == reserved

    second, _ = infer_gpt_with_kv_cache(model, prompt, cache)
    np.testing.assert_allclose(second, first, rtol=0.0, atol=0.0)
    assert [(id(buffer._k_storage), id(buffer._v_storage)) for buffer in cache._buffers] == pointers


def test_snapshot_is_independent_from_later_cache_reuse():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    snapshot = cache.snapshot()
    retained = [{"k": item["k"].copy(), "v": item["v"].copy()} for item in snapshot]

    cache.clear()
    infer_gpt_with_kv_cache(model, np.array([[7, 8]], dtype=np.int64), cache)

    for item, expected in zip(snapshot, retained):
        np.testing.assert_array_equal(item["k"], expected["k"])
        np.testing.assert_array_equal(item["v"], expected["v"])


def test_repr_and_byte_accounting_cover_all_layers():
    model = _model(layers=3)
    cache = GPTKVCache(model)
    assert "layers=3" in repr(cache)
    assert "uninitialized" in repr(cache)
    assert cache.storage_nbytes == 0
    assert cache.live_nbytes == 0

    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    assert cache.storage_nbytes == sum(buffer.storage_nbytes for buffer in cache._buffers)
    assert cache.live_nbytes == sum(buffer.live_nbytes for buffer in cache._buffers)
    assert "length=2" in repr(cache)
