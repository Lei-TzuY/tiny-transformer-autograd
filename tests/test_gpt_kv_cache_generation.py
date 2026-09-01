import numpy as np

from nn import GPT, GPTKVCache, generate_gpt_with_kv_cache


def _model(*, rope=False, context_len=6):
    np.random.seed(2201)
    return GPT(
        vocab_size=19,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        norm="rmsnorm" if rope else "layernorm",
        pos_encoding="rope" if rope else "learned",
        ffn="swiglu" if rope else "gelu",
    )


def test_greedy_generation_matches_historical_cached_generation_inside_window():
    model = _model(context_len=8)
    prompt = np.array([[1, 2, 3]], dtype=np.int64)

    expected = model.generate(prompt, 4, strategy="greedy", use_cache=True)
    actual, cache = generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        strategy="greedy",
    )

    np.testing.assert_array_equal(actual, expected)
    assert isinstance(cache, GPTKVCache)
    assert cache.length == 0 or cache.length <= model.context_len


def test_greedy_generation_matches_across_sliding_window_refills():
    model = _model(context_len=4)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)

    expected = model.generate(prompt, 6, strategy="greedy", use_cache=True)
    actual, cache = generate_gpt_with_kv_cache(
        model,
        prompt,
        6,
        strategy="greedy",
    )

    np.testing.assert_array_equal(actual, expected)
    assert cache.initialized
    assert cache.storage_nbytes > 0


def test_sampling_generation_preserves_historical_rng_trajectory():
    model = _model(context_len=7)
    prompt = np.array([[1, 2, 3]], dtype=np.int64)

    np.random.seed(9941)
    expected = model.generate(
        prompt,
        4,
        strategy="sample",
        temperature=0.8,
        top_k=7,
        top_p=0.9,
        use_cache=True,
    )
    expected_state = np.random.get_state()

    np.random.seed(9941)
    actual, _ = generate_gpt_with_kv_cache(
        model,
        prompt,
        4,
        strategy="sample",
        temperature=0.8,
        top_k=7,
        top_p=0.9,
    )
    actual_state = np.random.get_state()

    np.testing.assert_array_equal(actual, expected)
    assert actual_state[0] == expected_state[0]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]


def test_left_padded_rope_generation_matches_historical_cached_path():
    model = _model(rope=True, context_len=5)
    prompt = np.array([[0, 0, 2], [0, 4, 5]], dtype=np.int64)
    mask = np.array([[0, 0, 1], [0, 1, 1]], dtype=bool)

    expected = model.generate(
        prompt,
        5,
        strategy="greedy",
        use_cache=True,
        attention_mask=mask,
    )
    actual, _ = generate_gpt_with_kv_cache(
        model,
        prompt,
        5,
        strategy="greedy",
        attention_mask=mask,
    )
    np.testing.assert_array_equal(actual, expected)


def test_generation_reuses_preallocated_storage_across_window_refills():
    model = _model(context_len=4)
    cache = GPTKVCache(model)
    prompt = np.array([[1, 2, 3, 4]], dtype=np.int64)

    generated, returned = generate_gpt_with_kv_cache(
        model,
        prompt,
        5,
        strategy="greedy",
        cache=cache,
    )
    assert returned is cache
    assert generated.shape == (1, 9)
    assert cache.initialized
    # Every layer has exactly one fixed-capacity allocation despite several
    # context-window resets during generation.
    assert all(buffer.storage_nbytes > 0 for buffer in cache._buffers)


def test_zero_token_generation_leaves_new_cache_uninitialized():
    model = _model()
    cache = GPTKVCache(model)
    prompt = np.array([[1, 2]], dtype=np.int64)

    generated, returned = generate_gpt_with_kv_cache(
        model,
        prompt,
        0,
        strategy="greedy",
        cache=cache,
    )
    np.testing.assert_array_equal(generated, prompt)
    assert returned is cache
    assert not cache.initialized
    assert cache.length == 0
