import numpy as np

from nn import (
    GPT,
    beam_generate_gpt_with_kv_cache,
    infer_gpt_with_kv_cache,
)


def _model(*, rope=False, context_len=9):
    np.random.seed(1201)
    kwargs = {}
    if rope:
        kwargs.update(norm="rmsnorm", pos_encoding="rope", ffn="swiglu")
    return GPT(
        vocab_size=19,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        **kwargs,
    )


def _assert_cache_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_allclose(a["k"], b["k"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a["v"], b["v"], rtol=1e-12, atol=1e-12)


def test_buffered_beam_matches_legacy_learned_position_search():
    model = _model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)
    expected = model.generate(
        prompt,
        4,
        strategy="beam",
        beam_width=3,
        temperature=0.85,
        use_cache=True,
    )
    actual, cache = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
        temperature=0.85,
    )
    np.testing.assert_array_equal(actual, expected)
    assert cache.length == actual.shape[1]

    _, legacy_cache = model.infer(actual)
    _assert_cache_equal(cache.snapshot(), legacy_cache)


def test_buffered_beam_matches_rope_left_padded_legacy_search():
    model = _model(rope=True)
    prompt = np.array([[0, 0, 4, 5]], dtype=np.int64)
    mask = np.array([[0, 0, 1, 1]], dtype=bool)

    expected = model.generate(
        prompt,
        3,
        strategy="beam",
        beam_width=2,
        temperature=1.0,
        use_cache=True,
        attention_mask=mask,
    )
    actual, cache = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        3,
        beam_width=2,
        attention_mask=mask,
    )
    np.testing.assert_array_equal(actual, expected)
    assert cache.length == actual.shape[1]

    full_mask = np.concatenate(
        [mask, np.ones((1, 3), dtype=bool)],
        axis=1,
    )
    positions = np.maximum(np.cumsum(full_mask, axis=1) - 1, 0)
    _, legacy_cache = model.infer(
        actual,
        attention_mask=full_mask,
        position_ids=positions,
    )
    _assert_cache_equal(cache.snapshot(), legacy_cache)


def test_buffered_beam_strict_window_matches_legacy_and_returns_full_window_cache():
    model = _model(context_len=5)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)
    expected = model.generate(
        prompt,
        4,
        strategy="beam",
        beam_width=3,
        use_cache=True,
    )
    actual, cache = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
    )
    np.testing.assert_array_equal(actual, expected)
    assert cache.length == model.context_len

    _, legacy_cache = model.infer(actual[:, -model.context_len :])
    _assert_cache_equal(cache.snapshot(), legacy_cache)


def test_zero_token_beam_prefills_and_returns_prompt_aligned_cache():
    model = _model()
    prompt = np.array([[2, 4, 6]], dtype=np.int64)
    actual, cache = beam_generate_gpt_with_kv_cache(model, prompt, 0)
    np.testing.assert_array_equal(actual, prompt)
    assert cache.length == prompt.shape[1]
    _, legacy_cache = model.infer(prompt)
    _assert_cache_equal(cache.snapshot(), legacy_cache)


def test_returned_best_cache_can_continue_incremental_inference():
    model = _model(context_len=10)
    prompt = np.array([[3, 1]], dtype=np.int64)
    sequence, cache = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        3,
        beam_width=2,
    )
    token = np.array([[7]], dtype=np.int64)
    buffered_logits, _ = infer_gpt_with_kv_cache(model, token, cache)
    full = np.concatenate([sequence, token], axis=1)
    legacy_logits, legacy_cache = model.infer(full)

    np.testing.assert_allclose(
        buffered_logits[:, -1, :],
        legacy_logits[:, -1, :],
        rtol=1e-12,
        atol=1e-12,
    )
    _assert_cache_equal(cache.snapshot(), legacy_cache)
