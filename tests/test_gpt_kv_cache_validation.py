import numpy as np
import pytest

import nn.attention as attention_module
from nn import GPT, GPTKVCache, generate_gpt_with_kv_cache, infer_gpt_with_kv_cache


def _model(seed=4401):
    np.random.seed(seed)
    return GPT(
        vocab_size=13,
        context_len=6,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
    )


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_cache_constructor_and_public_type_validation():
    with pytest.raises(TypeError, match="model must be a GPT"):
        GPTKVCache(object())

    model = _model()
    cache = GPTKVCache(model)
    with pytest.raises(TypeError, match="model must be a GPT"):
        infer_gpt_with_kv_cache(object(), np.array([[1]], dtype=np.int64), cache)
    with pytest.raises(TypeError, match="cache must be a GPTKVCache"):
        infer_gpt_with_kv_cache(model, np.array([[1]], dtype=np.int64), object())


def test_cache_is_bound_to_exact_model_instance():
    first = _model(1)
    second = _model(1)
    cache = GPTKVCache(first)

    with pytest.raises(ValueError, match="different GPT instance"):
        infer_gpt_with_kv_cache(second, np.array([[1]], dtype=np.int64), cache)
    assert not cache.initialized


def test_attention_mask_and_position_validation_match_gpt_contract():
    model = _model()
    cache = GPTKVCache(model)
    tokens = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="attention_mask must have shape"):
        infer_gpt_with_kv_cache(
            model,
            tokens,
            cache,
            attention_mask=np.ones((1, 1), dtype=bool),
        )
    assert not cache.initialized

    with pytest.raises(ValueError, match="position_ids must have shape"):
        infer_gpt_with_kv_cache(
            model,
            tokens,
            cache,
            position_ids=np.array([[0]], dtype=np.int64),
        )
    assert not cache.initialized


def test_truncate_validation_and_zero_length_version_reset():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1, 2, 3]], dtype=np.int64), cache)

    for bad in (True, np.bool_(False), 1.5, "1"):
        with pytest.raises(TypeError):
            cache.truncate(bad)
    with pytest.raises(ValueError):
        cache.truncate(-1)
    with pytest.raises(ValueError):
        cache.truncate(4)

    cache.truncate(0)
    assert cache.length == 0
    assert cache._model_versions is None
    model.blocks[0].attn.W_v.weight.data[...] += 0.01
    infer_gpt_with_kv_cache(model, np.array([[4]], dtype=np.int64), cache)
    assert cache.length == 1


def test_generation_rejects_nonempty_or_wrong_cache_and_beam_strategy():
    model = _model()
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, np.array([[1]], dtype=np.int64), cache)

    with pytest.raises(ValueError, match="must be empty"):
        generate_gpt_with_kv_cache(
            model,
            np.array([[1]], dtype=np.int64),
            1,
            strategy="greedy",
            cache=cache,
        )
    with pytest.raises(TypeError, match="GPTKVCache or None"):
        generate_gpt_with_kv_cache(
            model,
            np.array([[1]], dtype=np.int64),
            1,
            strategy="greedy",
            cache=object(),
        )
    with pytest.raises(ValueError, match="sample.*greedy"):
        generate_gpt_with_kv_cache(
            model,
            np.array([[1]], dtype=np.int64),
            1,
            strategy="beam",
        )


def test_corrupted_partial_layer_initialization_fails_closed():
    model = _model()
    cache = GPTKVCache(model)
    key = np.zeros((1, 2, 1, 4), dtype=np.float64)
    cache._buffers[0].append(key, key)

    with pytest.raises(RuntimeError, match="partially initialized"):
        _ = cache.length
    with pytest.raises(RuntimeError, match="partially initialized"):
        infer_gpt_with_kv_cache(model, np.array([[1]], dtype=np.int64), cache)


def test_buffered_gpt_incremental_decode_does_not_use_attention_concatenate(monkeypatch):
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)
    token = np.array([[3]], dtype=np.int64)
    cache = GPTKVCache(model)
    infer_gpt_with_kv_cache(model, prompt, cache)
    _, legacy = model.infer(prompt)

    def forbidden(*args, **kwargs):
        raise AssertionError("attention concatenate used")

    monkeypatch.setattr(attention_module.np, "concatenate", forbidden)
    logits, _ = infer_gpt_with_kv_cache(model, token, cache)
    assert logits.shape == (1, 1, model.vocab_size)

    with pytest.raises(AssertionError, match="attention concatenate used"):
        model.infer(token, legacy)


def test_inference_is_numpy_rng_neutral_on_success_and_failure():
    model = _model()
    cache = GPTKVCache(model)
    np.random.seed(771)
    before = np.random.get_state()
    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    after = np.random.get_state()
    assert _rng_state_equal(before, after)

    before_failure = np.random.get_state()
    with pytest.raises(ValueError):
        infer_gpt_with_kv_cache(
            model,
            np.array([[3]], dtype=np.int64),
            cache,
            attention_mask=np.ones((1, 1), dtype=bool),
        )
    after_failure = np.random.get_state()
    assert _rng_state_equal(before_failure, after_failure)


def test_invalid_tensor_version_metadata_rejected_without_cache_write():
    model = _model()
    cache = GPTKVCache(model)
    victim = model.blocks[0].attn.W_q.weight
    original = victim._version
    victim._version = np.int64(original)
    try:
        with pytest.raises(RuntimeError, match="invalid mutation-version"):
            infer_gpt_with_kv_cache(model, np.array([[1]], dtype=np.int64), cache)
    finally:
        victim._version = original
    assert not cache.initialized
