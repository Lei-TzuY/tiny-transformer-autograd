import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from nn import GPT, convert_gpt_kv_heads


def _lora_model(kv_heads=4):
    np.random.seed(88)
    return GPT(
        vocab_size=17,
        context_len=5,
        d_model=8,
        num_heads=4,
        num_kv_heads=kv_heads,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        norm="rmsnorm",
        pos_encoding="rope",
        ffn="swiglu",
        lora_rank=2,
        lora_alpha=4.0,
    )


def _fill_head_rows(array, heads, head_dim, base):
    rows = array.reshape(heads, head_dim, *array.shape[1:])
    for head in range(heads):
        rows[head].fill(base + head)


def test_lora_compression_transforms_base_and_b_rows_but_keeps_a_exact():
    source = _lora_model(4)
    attention = source.blocks[0].attn
    _fill_head_rows(attention.W_k.weight.data, 4, 2, 10.0)
    _fill_head_rows(attention.W_v.weight.data, 4, 2, 20.0)
    _fill_head_rows(attention.W_k.lora_B.data, 4, 2, 30.0)
    _fill_head_rows(attention.W_v.lora_B.data, 4, 2, 40.0)
    k_a = attention.W_k.lora_A.data.copy()
    v_a = attention.W_v.lora_A.data.copy()

    converted = convert_gpt_kv_heads(source, 2)
    target = converted.blocks[0].attn

    assert converted.config()["num_kv_heads"] == 2
    assert converted.config()["lora_rank"] == 2
    assert converted.config()["lora_alpha"] == 4.0
    np.testing.assert_array_equal(target.W_k.lora_A.data, k_a)
    np.testing.assert_array_equal(target.W_v.lora_A.data, v_a)

    for tensor, base in (
        (target.W_k.weight.data, 10.0),
        (target.W_v.weight.data, 20.0),
        (target.W_k.lora_B.data, 30.0),
        (target.W_v.lora_B.data, 40.0),
    ):
        rows = tensor.reshape(2, 2, *tensor.shape[1:])
        np.testing.assert_array_equal(rows[0], np.full_like(rows[0], base + 0.5))
        np.testing.assert_array_equal(rows[1], np.full_like(rows[1], base + 2.5))


def test_lora_gqa_to_mha_expansion_preserves_effective_function():
    source = _lora_model(2)
    source.eval()
    converted = convert_gpt_kv_heads(source, 4)
    tokens = np.array([[1, 2, 4, 6]], dtype=np.int64)

    source_logits, _ = source.infer(tokens)
    converted_logits, _ = converted.infer(tokens)
    np.testing.assert_allclose(converted_logits, source_logits, rtol=1e-12, atol=1e-12)

    for projection_name in ("W_k", "W_v"):
        source_projection = getattr(source.blocks[0].attn, projection_name)
        target_projection = getattr(converted.blocks[0].attn, projection_name)
        compact_weight = source_projection.weight.data.reshape(2, 2, 8)
        expanded_weight = target_projection.weight.data.reshape(4, 2, 8)
        np.testing.assert_array_equal(expanded_weight, np.repeat(compact_weight, 2, axis=0))

        compact_b = source_projection.lora_B.data.reshape(2, 2, 2)
        expanded_b = target_projection.lora_B.data.reshape(4, 2, 2)
        np.testing.assert_array_equal(expanded_b, np.repeat(compact_b, 2, axis=0))
        np.testing.assert_array_equal(
            target_projection.lora_A.data, source_projection.lora_A.data
        )


def test_converted_lora_model_keeps_frozen_base_and_receives_adapter_gradients():
    source = _lora_model(4)
    converted = convert_gpt_kv_heads(source, 2)
    attention = converted.blocks[0].attn

    assert attention.W_k.weight.requires_grad is False
    assert attention.W_v.weight.requires_grad is False
    assert attention.W_k.lora_A.requires_grad is True
    assert attention.W_k.lora_B.requires_grad is True
    assert attention.W_v.lora_A.requires_grad is True
    assert attention.W_v.lora_B.requires_grad is True

    tokens = np.array([[1, 3, 5, 7]], dtype=np.int64)
    targets = np.array([[3, 5, 7, 9]], dtype=np.int64)
    loss = ops.cross_entropy(converted(tokens), targets)
    loss.backward()

    for adapter in (
        attention.W_k.lora_A,
        attention.W_k.lora_B,
        attention.W_v.lora_A,
        attention.W_v.lora_B,
    ):
        assert adapter.grad is not None
        assert np.isfinite(adapter.grad).all()
