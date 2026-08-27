"""Regression coverage for oversized real attention hyperparameters."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.attention import MultiHeadAttention, RotaryEmbedding, SelfAttention


def _rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_rope_base_overflow_is_normalized(value):
    with pytest.raises(ValueError, match="RoPE base must be finite"):
        RotaryEmbedding(4, 8, base=value)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_self_attention_dropout_overflow_is_normalized_before_rng(value):
    np.random.seed(123)
    before = np.random.get_state()

    with pytest.raises(ValueError, match="dropout must be finite"):
        SelfAttention(4, dropout=value)

    _rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_multihead_attention_dropout_overflow_is_normalized_before_rng(value):
    np.random.seed(456)
    before = np.random.get_state()

    with pytest.raises(ValueError, match="dropout must be finite"):
        MultiHeadAttention(4, 2, dropout=value)

    _rng_state_equal(np.random.get_state(), before)


def test_large_representable_rope_base_remains_supported():
    rope = RotaryEmbedding(4, 8, base=10**300)

    assert rope.base == float(10**300)
    assert np.isfinite(rope.cos).all()
    assert np.isfinite(rope.sin).all()


def test_large_representable_dropout_keeps_range_validation():
    with pytest.raises(ValueError, match="dropout must be less than 1.0"):
        SelfAttention(4, dropout=10**300)
