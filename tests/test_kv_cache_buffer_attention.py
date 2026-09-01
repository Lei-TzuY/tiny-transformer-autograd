"""Compatibility of KVCacheBuffer views with existing attention inference APIs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.attention import MultiHeadAttention, RotaryEmbedding, SelfAttention
from nn.kv_cache import KVCacheBuffer


def test_multi_head_attention_accepts_buffer_live_view_as_legacy_cache_mapping():
    np.random.seed(123)
    attention = MultiHeadAttention(8, 2)
    attention.eval()
    prefix = np.random.randn(1, 2, 8)
    token = np.random.randn(1, 1, 8)

    _, legacy = attention.infer(prefix)
    buffer = KVCacheBuffer(5)
    buffer.append(legacy["k"], legacy["v"])

    expected, expected_cache = attention.infer(token, cache=legacy)
    actual, actual_cache = attention.infer(token, cache=buffer.view())

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_cache["k"], expected_cache["k"])
    np.testing.assert_array_equal(actual_cache["v"], expected_cache["v"])


def test_single_head_attention_accepts_buffer_live_view():
    np.random.seed(456)
    attention = SelfAttention(6)
    attention.eval()
    prefix = np.random.randn(2, 2, 6)
    token = np.random.randn(2, 1, 6)

    _, legacy = attention.infer(prefix)
    buffer = KVCacheBuffer(4)
    buffer.append(legacy["k"], legacy["v"])

    expected, expected_cache = attention.infer(token, cache=legacy)
    actual, actual_cache = attention.infer(token, cache=buffer.view())

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_cache["k"], expected_cache["k"])
    np.testing.assert_array_equal(actual_cache["v"], expected_cache["v"])


def test_rope_multi_head_cached_decode_accepts_buffer_view():
    np.random.seed(789)
    rope = RotaryEmbedding(dim=4, max_pos=8)
    attention = MultiHeadAttention(8, 2, rope=rope)
    attention.eval()
    prefix = np.random.randn(1, 3, 8)
    token = np.random.randn(1, 1, 8)

    _, legacy = attention.infer(prefix)
    buffer = KVCacheBuffer(8)
    buffer.append(legacy["k"], legacy["v"])

    expected, _ = attention.infer(token, cache=legacy)
    actual, _ = attention.infer(token, cache=buffer.view())
    np.testing.assert_array_equal(actual, expected)


def test_buffer_can_absorb_returned_cache_then_continue_after_truncate():
    np.random.seed(321)
    attention = MultiHeadAttention(8, 2)
    attention.eval()
    prefix = np.random.randn(1, 2, 8)
    token_a = np.random.randn(1, 1, 8)
    token_b = np.random.randn(1, 1, 8)

    _, first_cache = attention.infer(prefix)
    buffer = KVCacheBuffer(6)
    buffer.append(first_cache["k"], first_cache["v"])

    _, full_a = attention.infer(token_a, cache=buffer.view())
    # Only append the newly produced tail, not the whole concatenated legacy cache.
    buffer.append(full_a["k"][..., -1:, :], full_a["v"][..., -1:, :])
    branch_point = buffer.snapshot()

    out_b, full_b = attention.infer(token_b, cache=buffer.view())

    buffer.truncate(2)
    buffer.append(full_a["k"][..., -1:, :], full_a["v"][..., -1:, :])
    replay_b, replay_cache = attention.infer(token_b, cache=buffer.view())

    np.testing.assert_array_equal(buffer.snapshot()["k"], branch_point["k"])
    np.testing.assert_array_equal(buffer.snapshot()["v"], branch_point["v"])
    np.testing.assert_array_equal(replay_b, out_b)
    np.testing.assert_array_equal(replay_cache["k"], full_b["k"])
    np.testing.assert_array_equal(replay_cache["v"], full_b["v"])
