import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import GPT, GroupedQueryAttention, MultiHeadAttention, convert_gpt_kv_heads


def _model(kv_heads=None, *, rope=False, lora_rank=0):
    np.random.seed(123)
    kwargs = {}
    if kv_heads is not None:
        kwargs["num_kv_heads"] = kv_heads
    return GPT(
        vocab_size=19,
        context_len=6,
        d_model=8,
        num_heads=4,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        pos_encoding="rope" if rope else "learned",
        norm="rmsnorm" if rope else "layernorm",
        ffn="swiglu" if rope else "gelu",
        lora_rank=lora_rank,
        lora_alpha=2.0,
        **kwargs,
    )


def _head_rows(attention, name):
    matrix = getattr(attention, name).weight.data
    return matrix.reshape(attention.num_kv_heads if hasattr(attention, "num_kv_heads") else attention.num_heads, 2, 8)


def _set_repeated_pairs(attention):
    for projection_name in ("W_k", "W_v"):
        projection = getattr(attention, projection_name)
        rows = projection.weight.data.reshape(4, 2, 8)
        first = np.arange(16, dtype=np.float64).reshape(2, 8) / 100.0
        second = -np.arange(16, dtype=np.float64).reshape(2, 8) / 80.0
        rows[0] = rows[1] = first
        rows[2] = rows[3] = second


def test_mha_to_gqa_averages_contiguous_kv_heads_and_keeps_other_state_exact():
    source = _model()
    block = source.blocks[0]
    assert isinstance(block.attn, MultiHeadAttention)

    for projection_name, offset in (("W_k", 10.0), ("W_v", 20.0)):
        rows = getattr(block.attn, projection_name).weight.data.reshape(4, 2, 8)
        for head in range(4):
            rows[head].fill(offset + head)

    source_state = source.state_dict()
    converted = convert_gpt_kv_heads(source, 2)

    assert converted is not source
    assert converted.num_kv_heads == 2
    assert converted.config()["num_kv_heads"] == 2
    assert all(isinstance(block.attn, GroupedQueryAttention) for block in converted.blocks)

    for projection_name, offset in (("W_k", 10.0), ("W_v", 20.0)):
        rows = getattr(converted.blocks[0].attn, projection_name).weight.data.reshape(2, 2, 8)
        np.testing.assert_array_equal(rows[0], np.full((2, 8), offset + 0.5))
        np.testing.assert_array_equal(rows[1], np.full((2, 8), offset + 2.5))

    converted_state = converted.state_dict()
    for name, value in source_state.items():
        if name.endswith((".attn.W_k.weight", ".attn.W_v.weight")):
            continue
        np.testing.assert_array_equal(converted_state[name], value)


def test_gqa_to_mha_expansion_preserves_forward_inference_and_expands_cache():
    source = _model(2, rope=True)
    source.eval()
    converted = convert_gpt_kv_heads(source, 4)

    assert converted.num_kv_heads == 4
    assert "num_kv_heads" not in converted.config()
    assert all(isinstance(block.attn, MultiHeadAttention) for block in converted.blocks)

    tokens = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=np.int64)
    source_graph = source(tokens).data
    converted_graph = converted(tokens).data
    np.testing.assert_allclose(converted_graph, source_graph, rtol=1e-12, atol=1e-12)

    source_logits, source_cache = source.infer(tokens)
    converted_logits, converted_cache = converted.infer(tokens)
    np.testing.assert_allclose(converted_logits, source_logits, rtol=1e-12, atol=1e-12)

    for compact, expanded in zip(source_cache, converted_cache):
        np.testing.assert_allclose(
            expanded["k"], np.repeat(compact["k"], 2, axis=1), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            expanded["v"], np.repeat(compact["v"], 2, axis=1), rtol=0.0, atol=0.0
        )


def test_mqa_to_gqa_expansion_is_function_preserving():
    source = _model(1)
    source.eval()
    converted = convert_gpt_kv_heads(source, 2)
    tokens = np.array([[1, 4, 2, 7, 3]], dtype=np.int64)

    source_logits, _ = source.infer(tokens)
    converted_logits, _ = converted.infer(tokens)
    np.testing.assert_allclose(converted_logits, source_logits, rtol=1e-12, atol=1e-12)


def test_mean_compression_is_exact_when_source_kv_heads_are_repeated_by_group():
    source = _model()
    for block in source.blocks:
        _set_repeated_pairs(block.attn)
    source.eval()

    converted = convert_gpt_kv_heads(source, 2)
    tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    source_logits, _ = source.infer(tokens)
    converted_logits, _ = converted.infer(tokens)

    np.testing.assert_allclose(converted_logits, source_logits, rtol=1e-12, atol=1e-12)


def test_same_head_count_returns_independent_exact_clone():
    source = _model(2)
    converted = convert_gpt_kv_heads(source, 2)

    source_state = source.state_dict()
    converted_state = converted.state_dict()
    assert source_state.keys() == converted_state.keys()
    for name in source_state:
        np.testing.assert_array_equal(converted_state[name], source_state[name])

    converted.blocks[0].attn.W_q.weight.data[0, 0] += 1.0
    assert not np.array_equal(
        converted.blocks[0].attn.W_q.weight.data,
        source.blocks[0].attn.W_q.weight.data,
    )


def test_conversion_preserves_runtime_mode_gradient_checkpoint_and_trainability():
    source = _model(2)
    source.eval()
    source.grad_checkpoint = True
    source.token_emb.weight.requires_grad = False
    source.blocks[0].attn.W_q.weight.requires_grad = False

    converted = convert_gpt_kv_heads(source, 1)

    assert converted.training is False
    assert all(module.training is False for module in converted.modules())
    assert converted.grad_checkpoint is True
    target_tensors = dict(converted.named_tensors())
    source_tensors = dict(source.named_tensors())
    assert target_tensors.keys() == source_tensors.keys()
    for name in source_tensors:
        assert target_tensors[name].requires_grad is source_tensors[name].requires_grad


def test_conversion_does_not_copy_or_mutate_live_gradients():
    source = _model(2)
    parameter = source.blocks[0].attn.W_q.weight
    parameter.grad = np.full_like(parameter.data, 3.0)
    original_grad = parameter.grad
    original_values = original_grad.copy()

    converted = convert_gpt_kv_heads(source, 1)

    assert parameter.grad is original_grad
    np.testing.assert_array_equal(parameter.grad, original_values)
    assert converted.blocks[0].attn.W_q.weight.grad is None
