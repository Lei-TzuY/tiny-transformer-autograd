import numpy as np
import pytest

from nn import GroupedQueryAttention, MultiHeadAttention, RotaryEmbedding


def _copy_mha_endpoint_weights(grouped, reference):
    reference.W_q.weight.data[...] = grouped.W_q.weight.data
    reference.W_k.weight.data[...] = grouped.W_k.weight.data
    reference.W_v.weight.data[...] = grouped.W_v.weight.data
    reference.out_proj.weight.data[...] = grouped.out_proj.weight.data
    reference.out_proj.bias.data[...] = grouped.out_proj.bias.data


def test_compact_cache_uses_kv_head_count_not_query_head_count():
    np.random.seed(701)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = np.random.randn(3, 5, 8)

    out, cache = attention.infer(x)

    assert out.shape == (3, 5, 8)
    assert cache["k"].shape == (3, 2, 5, 2)
    assert cache["v"].shape == (3, 2, 5, 2)


def test_mqa_cache_has_exactly_one_kv_head():
    np.random.seed(702)
    attention = GroupedQueryAttention(12, 6, num_kv_heads=1, dropout=0.0)
    _, cache = attention.infer(np.random.randn(2, 4, 12))

    assert cache["k"].shape == (2, 1, 4, 2)
    assert cache["v"].shape == (2, 1, 4, 2)


def test_cached_decode_matches_full_prefix_decode():
    np.random.seed(703)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    prefix = np.random.randn(2, 4, 8)
    next_token = np.random.randn(2, 1, 8)

    _, cache = attention.infer(prefix)
    cached, updated = attention.infer(next_token, cache=cache)
    full, _ = attention.infer(np.concatenate([prefix, next_token], axis=1))

    np.testing.assert_allclose(cached, full[:, -1:, :], rtol=0, atol=2e-14)
    assert updated["k"].shape == (2, 2, 5, 2)
    assert updated["v"].shape == (2, 2, 5, 2)


def test_cached_rope_decode_matches_full_prefix_decode():
    np.random.seed(704)
    rope = RotaryEmbedding(2, max_pos=10)
    attention = GroupedQueryAttention(
        8, 4, num_kv_heads=2, dropout=0.0, rope=rope
    )
    prefix = np.random.randn(1, 3, 8)
    next_token = np.random.randn(1, 1, 8)

    _, cache = attention.infer(prefix)
    cached, _ = attention.infer(next_token, cache=cache)
    full, _ = attention.infer(np.concatenate([prefix, next_token], axis=1))

    np.testing.assert_allclose(cached, full[:, -1:, :], rtol=0, atol=3e-14)


def test_explicit_rope_positions_work_with_compact_kv_heads():
    np.random.seed(705)
    rope = RotaryEmbedding(2, max_pos=12)
    attention = GroupedQueryAttention(
        8, 4, num_kv_heads=2, dropout=0.0, rope=rope
    )
    x = np.random.randn(2, 2, 8)
    positions = np.array([[[1, 2]], [[4, 5]]], dtype=np.int64)

    out, cache = attention.infer(x, positions=positions)

    assert out.shape == (2, 2, 8)
    assert cache["k"].shape == (2, 2, 2, 2)
    assert np.isfinite(out).all()


def test_key_bias_broadcasts_over_query_heads_not_compact_kv_heads():
    np.random.seed(706)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = np.random.randn(2, 3, 8)
    key_bias = np.zeros((2, 3), dtype=np.float64)
    key_bias[0, 0] = -np.inf

    out, _ = attention.infer(x, key_bias=key_bias[:, None, None, :])

    assert out.shape == (2, 3, 8)
    assert np.isfinite(out).all()


def test_input_cache_is_not_mutated_when_appending_new_tokens():
    np.random.seed(707)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    _, cache = attention.infer(np.random.randn(1, 2, 8))
    old_k = cache["k"].copy()
    old_v = cache["v"].copy()

    _, updated = attention.infer(np.random.randn(1, 1, 8), cache=cache)

    np.testing.assert_array_equal(cache["k"], old_k)
    np.testing.assert_array_equal(cache["v"], old_v)
    assert updated["k"] is not cache["k"]
    assert updated["v"] is not cache["v"]


def test_cache_rejects_query_head_count_instead_of_kv_head_count():
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    bad = {
        "k": np.zeros((1, 4, 2, 2)),
        "v": np.zeros((1, 4, 2, 2)),
    }

    with pytest.raises(ValueError, match="head count must be 2"):
        attention.infer(np.zeros((1, 1, 8)), cache=bad)


def test_cache_rejects_nonfinite_compact_state():
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    bad = {
        "k": np.zeros((1, 2, 2, 2)),
        "v": np.zeros((1, 2, 2, 2)),
    }
    bad["k"][0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="only finite"):
        attention.infer(np.zeros((1, 1, 8)), cache=bad)


def test_mha_endpoint_inference_and_cache_match_existing_multihead_attention():
    np.random.seed(708)
    grouped = GroupedQueryAttention(8, 4, num_kv_heads=4, dropout=0.0)
    reference = MultiHeadAttention(8, 4, dropout=0.0)
    _copy_mha_endpoint_weights(grouped, reference)
    x = np.random.randn(2, 3, 8)

    grouped_out, grouped_cache = grouped.infer(x)
    reference_out, reference_cache = reference.infer(x)

    np.testing.assert_array_equal(grouped_out, reference_out)
    np.testing.assert_array_equal(grouped_cache["k"], reference_cache["k"])
    np.testing.assert_array_equal(grouped_cache["v"], reference_cache["v"])
