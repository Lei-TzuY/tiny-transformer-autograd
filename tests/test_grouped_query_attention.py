import numpy as np

import engine.ops as ops
from engine.tensor import Tensor
from nn import GroupedQueryAttention, MultiHeadAttention, RotaryEmbedding


def _copy_common_weights(source, target):
    target.W_q.weight.data[...] = source.W_q.weight.data
    target.out_proj.weight.data[...] = source.out_proj.weight.data
    target.out_proj.bias.data[...] = source.out_proj.bias.data


def _expand_kv_weight(weight, query_heads, kv_heads, head_dim):
    group = query_heads // kv_heads
    blocks = np.asarray(weight).reshape(kv_heads, head_dim, weight.shape[1])
    return np.repeat(blocks, group, axis=0).reshape(
        query_heads * head_dim, weight.shape[1]
    )


def _sum_expanded_kv_grad(gradient, query_heads, kv_heads, head_dim):
    group = query_heads // kv_heads
    blocks = np.asarray(gradient).reshape(
        kv_heads, group, head_dim, gradient.shape[1]
    )
    return blocks.sum(axis=1).reshape(kv_heads * head_dim, gradient.shape[1])


def test_mha_endpoint_matches_existing_multihead_attention_forward_and_vjp():
    np.random.seed(101)
    grouped = GroupedQueryAttention(8, 4, num_kv_heads=4, dropout=0.0)
    reference = MultiHeadAttention(8, 4, dropout=0.0)

    _copy_common_weights(grouped, reference)
    reference.W_k.weight.data[...] = grouped.W_k.weight.data
    reference.W_v.weight.data[...] = grouped.W_v.weight.data

    values = np.random.randn(2, 3, 8)
    left = Tensor(values, requires_grad=True)
    right = Tensor(values, requires_grad=True)

    grouped_out = grouped(left)
    reference_out = reference(right)
    np.testing.assert_array_equal(grouped_out.data, reference_out.data)

    seed = np.random.randn(*grouped_out.shape)
    grouped_out.backward(seed)
    reference_out.backward(seed)

    np.testing.assert_array_equal(left.grad, right.grad)
    np.testing.assert_array_equal(grouped.W_q.weight.grad, reference.W_q.weight.grad)
    np.testing.assert_array_equal(grouped.W_k.weight.grad, reference.W_k.weight.grad)
    np.testing.assert_array_equal(grouped.W_v.weight.grad, reference.W_v.weight.grad)
    np.testing.assert_array_equal(
        grouped.out_proj.weight.grad, reference.out_proj.weight.grad
    )
    np.testing.assert_array_equal(
        grouped.out_proj.bias.grad, reference.out_proj.bias.grad
    )


def test_true_gqa_matches_explicitly_expanded_multihead_reference_and_shared_vjp():
    np.random.seed(202)
    query_heads = 4
    kv_heads = 2
    head_dim = 2
    grouped = GroupedQueryAttention(
        8, query_heads, num_kv_heads=kv_heads, dropout=0.0
    )
    reference = MultiHeadAttention(8, query_heads, dropout=0.0)

    _copy_common_weights(grouped, reference)
    reference.W_k.weight.data[...] = _expand_kv_weight(
        grouped.W_k.weight.data, query_heads, kv_heads, head_dim
    )
    reference.W_v.weight.data[...] = _expand_kv_weight(
        grouped.W_v.weight.data, query_heads, kv_heads, head_dim
    )

    values = np.random.randn(2, 4, 8)
    left = Tensor(values, requires_grad=True)
    right = Tensor(values, requires_grad=True)

    grouped_out = grouped(left)
    reference_out = reference(right)
    np.testing.assert_allclose(grouped_out.data, reference_out.data, rtol=0, atol=1e-14)

    seed = np.random.randn(*grouped_out.shape)
    grouped_out.backward(seed)
    reference_out.backward(seed)

    np.testing.assert_allclose(left.grad, right.grad, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        grouped.W_q.weight.grad, reference.W_q.weight.grad, rtol=0, atol=2e-13
    )
    np.testing.assert_allclose(
        grouped.out_proj.weight.grad,
        reference.out_proj.weight.grad,
        rtol=0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        grouped.out_proj.bias.grad,
        reference.out_proj.bias.grad,
        rtol=0,
        atol=2e-13,
    )

    expected_k = _sum_expanded_kv_grad(
        reference.W_k.weight.grad, query_heads, kv_heads, head_dim
    )
    expected_v = _sum_expanded_kv_grad(
        reference.W_v.weight.grad, query_heads, kv_heads, head_dim
    )
    np.testing.assert_allclose(grouped.W_k.weight.grad, expected_k, rtol=0, atol=2e-13)
    np.testing.assert_allclose(grouped.W_v.weight.grad, expected_v, rtol=0, atol=2e-13)


def test_mqa_is_single_kv_head_and_reduces_projection_state():
    attention = GroupedQueryAttention(12, 6, num_kv_heads=1)

    assert attention.d_k == 2
    assert attention.group_size == 6
    assert attention.kv_width == 2
    assert attention.W_q.weight.shape == (12, 12)
    assert attention.W_k.weight.shape == (2, 12)
    assert attention.W_v.weight.shape == (2, 12)
    assert attention.out_proj.weight.shape == (12, 12)


def test_custom_batch_mask_and_fully_masked_row_are_supported():
    np.random.seed(303)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = Tensor(np.random.randn(2, 3, 8), requires_grad=True)
    mask = np.zeros((2, 3, 3), dtype=np.float64)
    mask[0, 1, :] = -np.inf

    out = attention(x, mask)
    assert out.shape == (2, 3, 8)
    assert np.isfinite(out.data).all()

    out.backward(np.ones_like(out.data))
    assert np.isfinite(x.grad).all()


def test_rope_gqa_forward_matches_expanded_mha_reference():
    np.random.seed(404)
    rope = RotaryEmbedding(dim=2, max_pos=8)
    grouped = GroupedQueryAttention(
        8, 4, num_kv_heads=2, dropout=0.0, rope=rope
    )
    reference = MultiHeadAttention(8, 4, dropout=0.0, rope=rope)
    _copy_common_weights(grouped, reference)
    reference.W_k.weight.data[...] = _expand_kv_weight(
        grouped.W_k.weight.data, 4, 2, 2
    )
    reference.W_v.weight.data[...] = _expand_kv_weight(
        grouped.W_v.weight.data, 4, 2, 2
    )

    x = np.random.randn(1, 5, 8)
    grouped_out = grouped(Tensor(x))
    reference_out = reference(Tensor(x))
    np.testing.assert_allclose(grouped_out.data, reference_out.data, rtol=0, atol=2e-14)


def test_grouped_attention_does_not_consume_rng_when_dropout_is_zero():
    np.random.seed(505)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = Tensor(np.arange(16, dtype=np.float64).reshape(1, 2, 8))
    before = np.random.get_state()

    attention(x)

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_grouped_attention_can_participate_in_larger_graph():
    np.random.seed(606)
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2, dropout=0.0)
    x = Tensor(np.random.randn(1, 3, 8), requires_grad=True)

    loss = ops.sum(attention(x) * attention(x))
    loss.backward()

    assert x.grad.shape == x.shape
    assert np.isfinite(x.grad).all()
    for gradient in (
        attention.W_q.weight.grad,
        attention.W_k.weight.grad,
        attention.W_v.weight.grad,
        attention.out_proj.weight.grad,
    ):
        assert gradient is not None
        assert np.isfinite(gradient).all()
        assert np.any(gradient != 0.0)
