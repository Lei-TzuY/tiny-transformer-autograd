import numpy as np
import pytest

from nn.streaming import stream_generate
from nn.transformer import GPT


def _model(*, pos_encoding="learned", num_layers=2, context_len=6):
    np.random.seed(5151)
    return GPT(
        vocab_size=23,
        context_len=context_len,
        d_model=8,
        num_heads=4,
        num_kv_heads=2,
        d_ff=16,
        num_layers=num_layers,
        dropout=0.0,
        norm="rmsnorm" if pos_encoding == "rope" else "layernorm",
        pos_encoding=pos_encoding,
        ffn="swiglu" if pos_encoding == "rope" else "gelu",
    )


def test_gpt_infer_returns_compact_per_layer_gqa_cache():
    model = _model(pos_encoding="rope")
    tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    logits, cache = model.infer(tokens)

    assert logits.shape == (2, 3, model.vocab_size)
    assert len(cache) == model.num_layers
    for entry in cache:
        assert entry["k"].shape == (2, 2, 3, 2)
        assert entry["v"].shape == (2, 2, 3, 2)


def test_gpt_cached_gqa_decode_matches_full_prefix_inference():
    model = _model(pos_encoding="rope")
    prompt = np.array([[1, 3, 5]], dtype=np.int64)
    next_token = np.array([[7]], dtype=np.int64)

    _, cache = model.infer(prompt)
    cached, next_cache = model.infer(next_token, cache)
    full, full_cache = model.infer(np.concatenate([prompt, next_token], axis=1))

    np.testing.assert_allclose(cached[:, -1, :], full[:, -1, :], rtol=0, atol=1e-12)
    for cached_entry, full_entry in zip(next_cache, full_cache):
        assert cached_entry["k"].shape[1] == 2
        np.testing.assert_allclose(cached_entry["k"], full_entry["k"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(cached_entry["v"], full_entry["v"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("pos_encoding", ["learned", "rope"])
def test_gqa_greedy_generation_matches_with_and_without_cache(pos_encoding):
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    cached = _model(pos_encoding=pos_encoding).generate(
        prompt,
        5,
        strategy="greedy",
        use_cache=True,
    )
    uncached = _model(pos_encoding=pos_encoding).generate(
        prompt,
        5,
        strategy="greedy",
        use_cache=False,
    )
    np.testing.assert_array_equal(cached, uncached)


def test_left_padded_batched_gqa_generation_matches_cached_and_uncached_paths():
    prompt = np.array(
        [
            [0, 0, 1, 2],
            [0, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    cached = _model(pos_encoding="rope").generate(
        prompt,
        4,
        strategy="greedy",
        use_cache=True,
        attention_mask=mask,
    )
    uncached = _model(pos_encoding="rope").generate(
        prompt,
        4,
        strategy="greedy",
        use_cache=False,
        attention_mask=mask,
    )
    np.testing.assert_array_equal(cached, uncached)


def test_gqa_model_rejects_query_head_count_cache_instead_of_compact_kv_count():
    model = _model(pos_encoding="rope")
    bad_entry = {
        "k": np.zeros((1, model.num_heads, 2, 2), dtype=np.float64),
        "v": np.zeros((1, model.num_heads, 2, 2), dtype=np.float64),
    }
    cache = [
        {"k": bad_entry["k"].copy(), "v": bad_entry["v"].copy()}
        for _ in range(model.num_layers)
    ]

    with pytest.raises(ValueError, match="head count must be 2"):
        model.infer(np.array([[1]], dtype=np.int64), cache)


def test_one_block_streaming_gqa_matches_strict_window_generation():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    strict = _model(pos_encoding="rope", num_layers=1, context_len=4).generate(
        prompt,
        8,
        strategy="greedy",
    )
    streamed = stream_generate(
        _model(pos_encoding="rope", num_layers=1, context_len=4),
        prompt,
        8,
        strategy="greedy",
    )
    np.testing.assert_array_equal(streamed, strict)


def test_one_block_masked_streaming_gqa_matches_strict_window_generation():
    prompt = np.array(
        [
            [0, 1, 3, 5],
            [0, 0, 2, 4],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    strict = _model(pos_encoding="rope", num_layers=1, context_len=4).generate(
        prompt,
        6,
        strategy="greedy",
        attention_mask=mask,
    )
    streamed = stream_generate(
        _model(pos_encoding="rope", num_layers=1, context_len=4),
        prompt,
        6,
        strategy="greedy",
        attention_mask=mask,
    )
    np.testing.assert_array_equal(streamed, strict)
