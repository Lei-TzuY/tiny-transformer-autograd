"""Regression tests for attention and RoPE public hyperparameters."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.attention import MultiHeadAttention, RotaryEmbedding, SelfAttention


@pytest.mark.parametrize("field", ["dim", "max_pos"])
@pytest.mark.parametrize(
    "bad_value",
    [0, -1, True, np.bool_(False), 1.5, np.float64(4.0), "4"],
)
def test_rope_dimensions_require_positive_nonboolean_integers(field, bad_value):
    kwargs = {"dim": 4, "max_pos": 8}
    kwargs[field] = bad_value

    with pytest.raises((TypeError, ValueError), match="RoPE|positive|integer"):
        RotaryEmbedding(**kwargs)


def test_rope_requires_an_even_head_dimension():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(3, 8)


@pytest.mark.parametrize(
    "bad_base",
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, np.bool_(False), "10000"],
)
def test_rope_base_must_be_finite_positive_real(bad_base):
    with pytest.raises((TypeError, ValueError), match="RoPE base"):
        RotaryEmbedding(4, 8, base=bad_base)


def test_rope_accepts_numpy_scalars_and_normalizes_them():
    rope = RotaryEmbedding(np.int32(4), np.int64(6), base=np.float32(10000.0))

    assert type(rope.dim) is int
    assert type(rope.max_pos) is int
    assert type(rope.base) is float
    assert rope.cos.shape == (6, 4)
    assert rope.sin.shape == (6, 4)
    assert np.isfinite(rope.cos).all()
    assert np.isfinite(rope.sin).all()


@pytest.mark.parametrize(
    "bad_offset",
    [True, np.bool_(False), 1.5, np.float64(1.0), "1"],
)
def test_rope_offset_rejects_noninteger_and_boolean_values(bad_offset):
    rope = RotaryEmbedding(4, 8)
    values = np.ones((1, 2, 4))

    with pytest.raises(TypeError, match="offset"):
        rope.rotate_np(values, offset=bad_offset)


def test_rope_offset_rejects_negative_and_accepts_numpy_integer():
    rope = RotaryEmbedding(4, 8)
    values = np.ones((1, 2, 4))

    with pytest.raises(ValueError, match="offset"):
        rope.rotate_np(values, offset=-1)

    actual = rope.rotate_np(values, offset=np.int64(2))
    expected = rope.rotate_np(values, offset=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("attention_cls", [SelfAttention, MultiHeadAttention])
@pytest.mark.parametrize(
    "bad_d_model",
    [0, -1, True, np.bool_(False), 4.5, np.float64(4.0), "4"],
)
def test_attention_d_model_requires_a_positive_integer(attention_cls, bad_d_model):
    args = (bad_d_model,) if attention_cls is SelfAttention else (bad_d_model, 2)
    with pytest.raises((TypeError, ValueError), match="d_model"):
        attention_cls(*args)


@pytest.mark.parametrize(
    "bad_heads",
    [0, -1, True, np.bool_(False), 1.5, np.float64(2.0), "2"],
)
def test_multihead_num_heads_requires_a_positive_integer(bad_heads):
    with pytest.raises((TypeError, ValueError), match="num_heads"):
        MultiHeadAttention(8, bad_heads)


@pytest.mark.parametrize("attention_cls", [SelfAttention, MultiHeadAttention])
@pytest.mark.parametrize(
    "bad_dropout",
    [np.nan, np.inf, -np.inf, -0.1, 1.0, True, np.bool_(False), "0.1"],
)
def test_attention_dropout_requires_a_finite_real_probability(
    attention_cls, bad_dropout
):
    args = (8,) if attention_cls is SelfAttention else (8, 2)
    with pytest.raises((TypeError, ValueError), match="dropout"):
        attention_cls(*args, dropout=bad_dropout)


def test_multihead_rejects_invalid_rope_type_before_parameter_allocation():
    np.random.seed(91)
    before = np.random.get_state()

    with pytest.raises(TypeError, match="RotaryEmbedding"):
        MultiHeadAttention(8, 2, rope=object())

    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_multihead_rejects_mismatched_rope_dimension_before_parameter_allocation():
    rope = RotaryEmbedding(2, 8)
    np.random.seed(92)
    before = np.random.get_state()

    with pytest.raises(ValueError, match="head dimension"):
        MultiHeadAttention(8, 2, rope=rope)

    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_attention_accepts_numpy_scalar_hyperparameters_and_runs():
    rope = RotaryEmbedding(np.int64(4), np.int64(4), base=np.float64(10000.0))
    attention = MultiHeadAttention(
        np.int64(8),
        np.int32(2),
        dropout=np.float32(0.0),
        rope=rope,
    )
    x = Tensor(np.arange(16.0).reshape(1, 2, 8) / 10.0)

    graph = attention(x).data
    inferred, _ = attention.infer(x.data)

    assert type(attention.d_model) is int
    assert type(attention.num_heads) is int
    assert type(attention.attn_drop.p) is float
    np.testing.assert_allclose(graph, inferred, atol=1e-12, rtol=1e-12)
