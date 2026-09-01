import numpy as np
import pytest

from nn import GPT, PersistentGPTKVCache, infer_gpt_with_persistent_kv_cache


def _model():
    np.random.seed(1503)
    return GPT(
        vocab_size=19,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def _entry(cache):
    return (
        [(layer.head, layer.length, layer.segment_count) for layer in cache._layers],
        cache._model_versions,
    )


def _assert_entry(cache, entry):
    layers, versions = entry
    assert cache._model_versions == versions
    for layer, (head, length, count) in zip(cache._layers, layers):
        assert layer.head is head
        assert layer.length == length
        assert layer.segment_count == count


def test_late_second_block_failure_restores_all_layer_heads(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    entry = _entry(cache)

    def fail(_):
        raise RuntimeError("late second block failure")

    monkeypatch.setattr(model.blocks[1].ff, "infer", fail)
    with pytest.raises(RuntimeError, match="late second block failure"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
        )

    _assert_entry(cache, entry)
    assert cache.length == 2
    assert cache.segment_count == 1


def test_final_head_failure_restores_every_new_segment(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    entry = _entry(cache)

    def fail(_):
        raise RuntimeError("final head failure")

    monkeypatch.setattr(model.head, "infer", fail)
    with pytest.raises(RuntimeError, match="final head failure"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
        )
    _assert_entry(cache, entry)


def test_first_call_final_failure_returns_to_exact_empty_state(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)

    def fail(_):
        raise RuntimeError("first call head failure")

    monkeypatch.setattr(model.head, "infer", fail)
    with pytest.raises(RuntimeError, match="first call head failure"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[1, 2]], dtype=np.int64),
            cache,
        )
    assert cache.length == 0
    assert cache.segment_count == 0
    assert cache.snapshot() is None
    assert cache._model_versions is None
    for layer in cache._layers:
        assert layer.head is None
        assert layer.key_layout is None
        assert layer.value_layout is None
        assert layer.key_dtype is None
        assert layer.value_dtype is None


def test_model_mutation_during_call_rolls_back_cache(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    entry = _entry(cache)
    original = model.head.infer

    def mutate_then_infer(x):
        model.token_emb.weight.data[0, 0] += 0.125
        return original(x)

    monkeypatch.setattr(model.head, "infer", mutate_then_infer)
    with pytest.raises(RuntimeError, match="model tensors changed during"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
        )
    _assert_entry(cache, entry)


def test_stale_model_rejected_before_token_projection(monkeypatch):
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    model.head.weight.data[0, 0] += 0.25
    called = []
    original = model.token_emb.infer

    def probe(ids):
        called.append(True)
        return original(ids)

    monkeypatch.setattr(model.token_emb, "infer", probe)
    with pytest.raises(RuntimeError, match="model tensors changed"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
        )
    assert called == []
    assert cache.length == 2


def test_clear_after_model_update_allows_clean_refill():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    model.head.weight.data[0, 0] += 0.25
    cache.clear()
    logits, _ = infer_gpt_with_persistent_kv_cache(
        model,
        np.array([[3, 4]], dtype=np.int64),
        cache,
    )
    expected, _ = model.infer(np.array([[3, 4]], dtype=np.int64))
    np.testing.assert_allclose(logits, expected, rtol=1e-12, atol=1e-12)
    assert cache.length == 2
    assert cache.segment_count == 1


def test_context_overflow_fails_before_cache_mutation():
    model = _model()
    cache = PersistentGPTKVCache(model)
    infer_gpt_with_persistent_kv_cache(
        model,
        np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        cache,
    )
    entry = _entry(cache)
    with pytest.raises(ValueError, match="exceed context_len"):
        infer_gpt_with_persistent_kv_cache(
            model,
            np.array([[8, 9]], dtype=np.int64),
            cache,
        )
    _assert_entry(cache, entry)
