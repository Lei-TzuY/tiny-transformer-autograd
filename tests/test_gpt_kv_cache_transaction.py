import numpy as np
import pytest

from nn import GPT, GPTKVCache, infer_gpt_with_kv_cache


def _model(*, layers=3):
    np.random.seed(3301)
    return GPT(
        vocab_size=17,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=layers,
        dropout=0.0,
    )


def _snapshot(cache):
    state = cache.snapshot()
    if state is None:
        return None
    return [{"k": item["k"].copy(), "v": item["v"].copy()} for item in state]


def _assert_snapshot_equal(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left["k"], right["k"])
        np.testing.assert_array_equal(left["v"], right["v"])


def test_first_call_late_block_failure_restores_all_layers_to_uninitialized():
    model = _model()
    cache = GPTKVCache(model)
    original = model.blocks[1].ff.infer

    def fail(_):
        raise RuntimeError("late block failure")

    model.blocks[1].ff.infer = fail
    try:
        with pytest.raises(RuntimeError, match="late block failure"):
            infer_gpt_with_kv_cache(model, np.array([[1, 2, 3]], dtype=np.int64), cache)
    finally:
        model.blocks[1].ff.infer = original

    assert cache.length == 0
    assert cache.initialized is False
    assert cache.storage_nbytes == 0
    assert cache.snapshot() is None
    assert all(not buffer.initialized for buffer in cache._buffers)


def test_incremental_late_failure_rolls_every_layer_back_to_entry_snapshot():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2, 3]], dtype=np.int64), cache)
    expected = _snapshot(cache)
    pointers = [(id(b._k_storage), id(b._v_storage)) for b in cache._buffers]

    original = model.blocks[-1].ff.infer

    def fail(_):
        raise RuntimeError("last block failure")

    model.blocks[-1].ff.infer = fail
    try:
        with pytest.raises(RuntimeError, match="last block failure"):
            infer_gpt_with_kv_cache(model, np.array([[4]], dtype=np.int64), cache)
    finally:
        model.blocks[-1].ff.infer = original

    assert cache.length == 3
    _assert_snapshot_equal(cache.snapshot(), expected)
    assert [(id(b._k_storage), id(b._v_storage)) for b in cache._buffers] == pointers


def test_final_lm_head_failure_rolls_back_all_layer_appends():
    model = _model(layers=2)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    expected = _snapshot(cache)
    original = model.head.infer

    def fail(_):
        raise RuntimeError("head failure")

    model.head.infer = fail
    try:
        with pytest.raises(RuntimeError, match="head failure"):
            infer_gpt_with_kv_cache(model, np.array([[3]], dtype=np.int64), cache)
    finally:
        model.head.infer = original

    assert cache.length == 2
    _assert_snapshot_equal(cache.snapshot(), expected)


def test_model_mutation_between_calls_rejects_stale_cache_before_projection():
    model = _model(layers=2)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    expected = _snapshot(cache)

    model.blocks[0].attn.W_k.weight.data[...] += 0.25
    original = model.token_emb.infer
    calls = []

    def observed(idx):
        calls.append(True)
        return original(idx)

    model.token_emb.infer = observed
    try:
        with pytest.raises(RuntimeError, match="model tensors changed"):
            infer_gpt_with_kv_cache(model, np.array([[3]], dtype=np.int64), cache)
    finally:
        model.token_emb.infer = original

    assert calls == []
    assert cache.length == 2
    _assert_snapshot_equal(cache.snapshot(), expected)


def test_model_mutation_during_call_rolls_cache_back():
    model = _model(layers=2)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    expected = _snapshot(cache)
    original = model.blocks[1].ff.infer
    victim = model.blocks[0].attn.W_q.weight

    def mutate(x):
        result = original(x)
        victim.data[...] += 1e-6
        return result

    model.blocks[1].ff.infer = mutate
    try:
        with pytest.raises(RuntimeError, match="changed during"):
            infer_gpt_with_kv_cache(model, np.array([[3]], dtype=np.int64), cache)
    finally:
        model.blocks[1].ff.infer = original

    assert cache.length == 2
    _assert_snapshot_equal(cache.snapshot(), expected)


def test_clear_discards_stale_version_binding_and_allows_refill_after_model_update():
    model = _model(layers=2)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    cache.clear()
    model.blocks[0].attn.W_k.weight.data[...] += 0.1

    logits, _ = infer_gpt_with_kv_cache(model, np.array([[3, 4]], dtype=np.int64), cache)
    expected, _ = model.infer(np.array([[3, 4]], dtype=np.int64))
    np.testing.assert_allclose(logits, expected, rtol=0.0, atol=0.0)
    assert cache.length == 2


def test_context_overflow_fails_before_embedding_or_cache_write():
    model = _model(layers=2)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(
        model,
        np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        cache,
    )
    expected = _snapshot(cache)
    original = model.token_emb.infer
    calls = []

    def observed(idx):
        calls.append(True)
        return original(idx)

    model.token_emb.infer = observed
    try:
        with pytest.raises(ValueError, match="exceed context_len"):
            infer_gpt_with_kv_cache(
                model,
                np.array([[8, 9]], dtype=np.int64),
                cache,
            )
    finally:
        model.token_emb.infer = original

    assert calls == []
    assert cache.length == 7
    _assert_snapshot_equal(cache.snapshot(), expected)
