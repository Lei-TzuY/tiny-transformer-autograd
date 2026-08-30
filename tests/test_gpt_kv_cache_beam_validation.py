import numpy as np
import pytest

import nn.attention as attention_module
from nn import GPT, GPTKVCache, beam_generate_gpt_with_kv_cache, fork_gpt_kv_cache


def _model():
    np.random.seed(1301)
    return GPT(
        vocab_size=17,
        context_len=7,
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


def test_fork_requires_gpt_cache():
    with pytest.raises(TypeError, match="GPTKVCache"):
        fork_gpt_kv_cache(None)


def test_buffered_beam_validates_public_options():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(TypeError, match="GPT"):
        beam_generate_gpt_with_kv_cache(object(), prompt, 1)
    with pytest.raises(TypeError, match="non-negative integer"):
        beam_generate_gpt_with_kv_cache(model, prompt, True)
    with pytest.raises(ValueError, match="non-negative integer"):
        beam_generate_gpt_with_kv_cache(model, prompt, -1)
    with pytest.raises(TypeError, match="positive integer"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1, beam_width=np.bool_(True))
    with pytest.raises(ValueError, match="positive integer"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1, beam_width=0)
    with pytest.raises(TypeError, match="real number"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1, temperature=True)
    with pytest.raises(ValueError, match="finite"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1, temperature=np.inf)
    with pytest.raises(ValueError, match="positive"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1, temperature=0.0)


def test_buffered_beam_rejects_multirow_batch_explicitly():
    model = _model()
    prompt = np.array([[1, 2], [3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="batch size 1"):
        beam_generate_gpt_with_kv_cache(model, prompt, 1)


def test_buffered_beam_uses_normal_left_padding_validation():
    model = _model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    bad_mask = np.array([[1, 0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="left-padded"):
        beam_generate_gpt_with_kv_cache(
            model,
            prompt,
            1,
            attention_mask=bad_mask,
        )


def test_buffered_beam_does_not_use_attention_cache_concatenate(monkeypatch):
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    class AttentionNumpyProxy:
        def __getattr__(self, name):
            if name == "concatenate":
                raise AssertionError("legacy attention concatenate reached")
            return getattr(np, name)

    monkeypatch.setattr(attention_module, "np", AttentionNumpyProxy())
    tokens, cache = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        2,
        beam_width=2,
    )
    assert tokens.shape == (1, 4)
    assert cache.length == 4


def test_buffered_beam_preserves_numpy_rng_on_success():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)
    np.random.seed(9123)
    before = np.random.get_state()
    beam_generate_gpt_with_kv_cache(model, prompt, 2, beam_width=2)
    after = np.random.get_state()
    assert _rng_state_equal(before, after)


def test_stale_cache_fork_does_not_change_numpy_rng():
    model = _model()
    cache = GPTKVCache(model)
    from nn import infer_gpt_with_kv_cache

    infer_gpt_with_kv_cache(model, np.array([[1, 2]], dtype=np.int64), cache)
    model.head.weight.data[0, 0] += 0.5
    np.random.seed(4455)
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="model tensors changed"):
        fork_gpt_kv_cache(cache)
    after = np.random.get_state()
    assert _rng_state_equal(before, after)
