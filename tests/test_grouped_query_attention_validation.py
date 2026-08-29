import numpy as np
import pytest

from engine.grad_mode import no_grad
from engine.tensor import Tensor
from nn import GroupedQueryAttention, RotaryEmbedding


@pytest.mark.parametrize(
    "args, error",
    [
        ((True, 4), TypeError),
        ((0, 4), ValueError),
        ((8, True), TypeError),
        ((8, 0), ValueError),
    ],
)
def test_required_dimensions_are_positive_integers(args, error):
    with pytest.raises(error):
        GroupedQueryAttention(*args)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_num_kv_heads_validation(value):
    error = TypeError if isinstance(value, (bool, float)) else ValueError
    with pytest.raises(error):
        GroupedQueryAttention(8, 4, num_kv_heads=value)


def test_none_kv_heads_is_mha_endpoint():
    attention = GroupedQueryAttention(8, 4)
    assert attention.num_kv_heads == 4
    assert attention.group_size == 1
    assert attention.kv_width == 8


def test_model_width_must_be_divisible_by_query_heads():
    with pytest.raises(ValueError, match="d_model must be divisible"):
        GroupedQueryAttention(10, 4, num_kv_heads=2)


def test_query_heads_must_be_divisible_by_kv_heads():
    with pytest.raises(ValueError, match="num_query_heads must be divisible"):
        GroupedQueryAttention(12, 6, num_kv_heads=4)


@pytest.mark.parametrize("value", [True, np.nan, np.inf, -0.1, 1.0])
def test_dropout_validation(value):
    error = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(error):
        GroupedQueryAttention(8, 4, 2, dropout=value)


def test_rope_type_and_dimension_validation():
    with pytest.raises(TypeError, match="RotaryEmbedding"):
        GroupedQueryAttention(8, 4, 2, rope=object())
    with pytest.raises(ValueError, match="RoPE dimension"):
        GroupedQueryAttention(8, 4, 2, rope=RotaryEmbedding(4, 8))


def test_forward_requires_tensor_with_rank_three_and_matching_width():
    attention = GroupedQueryAttention(8, 4, 2)
    with pytest.raises(TypeError, match="must be a Tensor"):
        attention(np.zeros((1, 2, 8)))
    with pytest.raises(ValueError, match="shape"):
        attention(Tensor(np.zeros((2, 8))))
    with pytest.raises(ValueError, match="shape"):
        attention(Tensor(np.zeros((1, 2, 7))))


def test_forward_rejects_invalid_masks_before_attention_output():
    attention = GroupedQueryAttention(8, 4, 2, dropout=0.0)
    x = Tensor(np.zeros((1, 2, 8)))
    with pytest.raises(ValueError, match="does not broadcast"):
        attention(x, np.zeros((3, 3)))
    bad = np.zeros((2, 2), dtype=np.float64)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN and \+inf"):
        attention(x, bad)


def test_forward_under_no_grad_is_detached():
    attention = GroupedQueryAttention(8, 4, 2, dropout=0.0)
    x = Tensor(np.random.randn(1, 2, 8), requires_grad=True)
    with no_grad():
        out = attention(x)
    assert out.requires_grad is False
    assert out._children == ()


@pytest.mark.parametrize(
    "value, error",
    [
        (np.zeros((2, 8)), ValueError),
        (np.zeros((1, 2, 7)), ValueError),
        (np.ones((1, 2, 8), dtype=bool), TypeError),
        (np.ones((1, 2, 8), dtype=np.complex128), TypeError),
    ],
)
def test_infer_input_validation(value, error):
    attention = GroupedQueryAttention(8, 4, 2)
    with pytest.raises(error):
        attention.infer(value)


def test_infer_rejects_nonfinite_input():
    attention = GroupedQueryAttention(8, 4, 2)
    value = np.zeros((1, 2, 8))
    value[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="only finite"):
        attention.infer(value)


def test_infer_rejects_invalid_key_bias():
    attention = GroupedQueryAttention(8, 4, 2)
    x = np.zeros((1, 2, 8))
    bias = np.zeros((1, 1, 1, 2))
    bias[..., 0] = np.inf
    with pytest.raises(ValueError, match="NaN and \+inf"):
        attention.infer(x, key_bias=bias)


def test_invalid_cache_fails_before_projection_result_is_used():
    attention = GroupedQueryAttention(8, 4, 2)
    with pytest.raises(TypeError, match="dictionary"):
        attention.infer(np.zeros((1, 1, 8)), cache=[])
    with pytest.raises(ValueError, match="contain 'k' and 'v'"):
        attention.infer(np.zeros((1, 1, 8)), cache={"k": np.zeros((1, 2, 0, 2))})


def test_repr_records_query_and_kv_head_counts():
    attention = GroupedQueryAttention(12, 6, 2, dropout=0.25)
    text = repr(attention)
    assert "d_model=12" in text
    assert "query_heads=6" in text
    assert "kv_heads=2" in text
    assert "dropout=0.25" in text
