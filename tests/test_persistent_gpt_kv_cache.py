import faulthandler

import numpy as np

from nn import GPT, PersistentGPTKVCache, infer_gpt_with_persistent_kv_cache


# Temporary CI diagnostic: the normal complete suite finishes in seconds. If a
# persistent-cache regression deadlocks, dump every Python thread and terminate
# instead of consuming the workflow's six-hour timeout.
faulthandler.dump_traceback_later(30, exit=True)


def _model(*, rope=False, context_len=10):
    np.random.seed(1501)
    kwargs = {}
    if rope:
        kwargs.update(norm="rmsnorm", pos_encoding="rope", ffn="swiglu")
    return GPT(
        vocab_size=23,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        **kwargs,
    )


def _assert_cache_close(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left["k"], right["k"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(left["v"], right["v"], rtol=1e-12, atol=1e-12)


def test_persistent_cache_prefill_matches_legacy_inference():
    model = _model()
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    expected_logits, expected_cache = model.infer(prompt)

    cache = PersistentGPTKVCache(model)
    actual_logits, returned = infer_gpt_with_persistent_kv_cache(model, prompt, cache)

    assert returned is cache
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-12, atol=1e-12)
    _assert_cache_close(cache.snapshot(), expected_cache)
    assert cache.length == 4
    assert cache.segment_count == 1
    assert cache.initialized
    assert cache.storage_nbytes == cache.live_nbytes > 0


def test_persistent_cache_incremental_decode_matches_full_prefix():
    model = _model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    token1 = np.array([[4]], dtype=np.int64)
    token2 = np.array([[5]], dtype=np.int64)

    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, prompt, cache)
    logits1, _ = infer_gpt_with_persistent_kv_cache(model, token1, cache)
    logits2, _ = infer_gpt_with_persistent_kv_cache(model, token2, cache)

    full1 = np.concatenate([prompt, token1], axis=1)
    expected1, _ = model.infer(full1)
    full2 = np.concatenate([full1, token2], axis=1)
    expected2, expected_cache = model.infer(full2)

    np.testing.assert_allclose(logits1[:, -1], expected1[:, -1], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(logits2[:, -1], expected2[:, -1], rtol=1e-11, atol=1e-11)
    _assert_cache_close(cache.snapshot(), expected_cache)
    assert cache.length == 5
    assert cache.segment_count == 3


def test_persistent_cache_rope_left_padding_matches_legacy():
    model = _model(rope=True)
    prompt = np.array([[0, 0, 7, 8]], dtype=np.int64)
    mask = np.array([[0, 0, 1, 1]], dtype=bool)
    positions = np.array([[0, 0, 0, 1]], dtype=np.int64)

    expected_logits, expected_cache = model.infer(
        prompt,
        attention_mask=mask,
        position_ids=positions,
    )
    cache = PersistentGPTKVCache(model)
    actual_logits, _ = infer_gpt_with_persistent_kv_cache(
        model,
        prompt,
        cache,
        attention_mask=mask,
        position_ids=positions,
    )
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-12, atol=1e-12)
    _assert_cache_close(cache.snapshot(), expected_cache)

    token = np.array([[9]], dtype=np.int64)
    next_mask = np.array([[0, 0, 1, 1, 1]], dtype=bool)
    next_position = np.array([[2]], dtype=np.int64)
    actual_next, _ = infer_gpt_with_persistent_kv_cache(
        model,
        token,
        cache,
        attention_mask=next_mask,
        position_ids=next_position,
    )
    full = np.concatenate([prompt, token], axis=1)
    full_positions = np.array([[0, 0, 0, 1, 2]], dtype=np.int64)
    expected_next, expected_cache = model.infer(
        full,
        attention_mask=next_mask,
        position_ids=full_positions,
    )
    np.testing.assert_allclose(
        actual_next[:, -1], expected_next[:, -1], rtol=1e-11, atol=1e-11
    )
    _assert_cache_close(cache.snapshot(), expected_cache)


def test_snapshot_is_independent_and_clear_drops_only_this_branch():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(
        model,
        np.array([[2, 4, 6]], dtype=np.int64),
        cache,
    )
    snapshot = cache.snapshot()
    before = [
        {"k": entry["k"].copy(), "v": entry["v"].copy()}
        for entry in snapshot
    ]
    snapshot[0]["k"][...] = 123.0
    snapshot[0]["v"][...] = -123.0
    fresh = cache.snapshot()
    np.testing.assert_array_equal(fresh[0]["k"], before[0]["k"])
    np.testing.assert_array_equal(fresh[0]["v"], before[0]["v"])

    cache.clear()
    assert cache.length == 0
    assert cache.segment_count == 0
    assert not cache.initialized
    assert cache.snapshot() is None
    assert cache.live_nbytes == 0


def test_repr_and_len_reflect_persistent_state():
    model = _model()
    cache = PersistentGPTKVCache(model)
    assert len(cache) == 0
    assert "segments=0" in repr(cache)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    assert len(cache) == 2
    text = repr(cache)
    assert "length=2" in text
    assert "segments=1" in text
