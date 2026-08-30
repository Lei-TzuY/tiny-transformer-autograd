import numpy as np

from nn import (
    GPT,
    beam_generate_gpt_with_kv_cache,
    beam_generate_gpt_with_persistent_kv_cache,
    infer_gpt_with_persistent_kv_cache,
)


def _model(*, rope=False, context_len=9):
    np.random.seed(1504)
    kwargs = {}
    if rope:
        kwargs.update(norm="rmsnorm", pos_encoding="rope", ffn="swiglu")
    return GPT(
        vocab_size=21,
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
        np.testing.assert_allclose(left["k"], right["k"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(left["v"], right["v"], rtol=1e-11, atol=1e-11)


def test_persistent_beam_matches_eager_branch_beam_and_legacy():
    model = _model()
    prompt = np.array([[1, 2, 3]], dtype=np.int64)

    expected = model.generate(
        prompt,
        4,
        strategy="beam",
        beam_width=3,
        temperature=0.9,
        use_cache=True,
    )
    eager, _ = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
        temperature=0.9,
    )
    persistent, cache = beam_generate_gpt_with_persistent_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
        temperature=0.9,
    )

    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(persistent, expected)
    assert cache.length == persistent.shape[1]
    assert cache.segment_count == 5

    _, legacy_cache = model.infer(persistent)
    _assert_cache_close(cache.snapshot(), legacy_cache)


def test_persistent_beam_rope_left_padding_matches_eager_beam():
    model = _model(rope=True)
    prompt = np.array([[0, 0, 4, 5]], dtype=np.int64)
    mask = np.array([[0, 0, 1, 1]], dtype=bool)

    eager, _ = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        3,
        beam_width=2,
        attention_mask=mask,
    )
    persistent, cache = beam_generate_gpt_with_persistent_kv_cache(
        model,
        prompt,
        3,
        beam_width=2,
        attention_mask=mask,
    )
    np.testing.assert_array_equal(persistent, eager)
    assert cache.length == persistent.shape[1]

    full_mask = np.concatenate([mask, np.ones((1, 3), dtype=bool)], axis=1)
    positions = np.maximum(np.cumsum(full_mask, axis=1) - 1, 0)
    _, legacy_cache = model.infer(
        persistent,
        attention_mask=full_mask,
        position_ids=positions,
    )
    _assert_cache_close(cache.snapshot(), legacy_cache)


def test_persistent_beam_strict_window_refill_matches_eager_path():
    model = _model(context_len=5)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)

    eager, _ = beam_generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
    )
    persistent, cache = beam_generate_gpt_with_persistent_kv_cache(
        model,
        prompt,
        4,
        beam_width=3,
    )
    np.testing.assert_array_equal(persistent, eager)
    assert cache.length == model.context_len

    _, legacy_cache = model.infer(persistent[:, -model.context_len :])
    _assert_cache_close(cache.snapshot(), legacy_cache)


def test_zero_token_persistent_beam_prefills_prompt_cache():
    model = _model()
    prompt = np.array([[2, 5, 8]], dtype=np.int64)
    sequence, cache = beam_generate_gpt_with_persistent_kv_cache(model, prompt, 0)
    np.testing.assert_array_equal(sequence, prompt)
    assert cache.length == 3
    assert cache.segment_count == 1
    _, expected_cache = model.infer(prompt)
    _assert_cache_close(cache.snapshot(), expected_cache)


def test_persistent_beam_winning_cache_continues_incrementally():
    model = _model(context_len=10)
    prompt = np.array([[3, 1]], dtype=np.int64)
    sequence, cache = beam_generate_gpt_with_persistent_kv_cache(
        model,
        prompt,
        3,
        beam_width=2,
    )
    token = np.array([[7]], dtype=np.int64)
    actual, _ = infer_gpt_with_persistent_kv_cache(model, token, cache)
    full = np.concatenate([sequence, token], axis=1)
    expected, expected_cache = model.infer(full)
    np.testing.assert_allclose(actual[:, -1], expected[:, -1], rtol=1e-11, atol=1e-11)
    _assert_cache_close(cache.snapshot(), expected_cache)
