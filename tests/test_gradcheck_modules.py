"""Module-level examples for parameter-aware finite-difference gradcheck."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradcheck import gradcheck
from engine.tensor import Tensor
from nn.attention import MultiHeadAttention, RotaryEmbedding
from nn.layers import LayerNorm, Linear, RMSNorm
from nn.transformer import TransformerBlock


def test_gradcheck_linear_lora_parameters_only():
    np.random.seed(31)
    layer = Linear(2, 2)
    layer.enable_lora(rank=1, alpha=2.0)
    x = Tensor([[0.4, -0.7], [1.2, 0.3]])

    names = [name for name, _ in layer.named_parameters()]
    assert names == ["lora_A", "lora_B"]
    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=2e-6,
        rtol=2e-5,
    )


def test_gradcheck_layernorm_affine_parameters():
    layer = LayerNorm(3)
    x = Tensor([[0.2, -0.4, 1.1], [0.8, 0.3, -0.6]])

    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=3e-6,
        rtol=3e-5,
    )


def test_gradcheck_rmsnorm_scale_parameter():
    layer = RMSNorm(3)
    x = Tensor([[0.2, -0.4, 1.1], [0.8, 0.3, -0.6]])

    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=3e-6,
        rtol=3e-5,
    )


def test_gradcheck_causal_multihead_attention_composition():
    """Numerically verify the complete causal attention VJP, not just matmul."""
    np.random.seed(41)
    attention = MultiHeadAttention(d_model=2, num_heads=1, dropout=0.0)
    x = Tensor([[[0.2, -0.4], [0.7, 0.1]]])

    assert gradcheck(
        lambda value: attention(value),
        x,
        parameters=attention.named_parameters(),
        atol=8e-6,
        rtol=8e-5,
    )


def test_gradcheck_rope_rmsnorm_swiglu_transformer_block():
    """Exercise modern Transformer composition through one numerical oracle."""
    np.random.seed(43)
    rope = RotaryEmbedding(dim=2, max_pos=2)
    block = TransformerBlock(
        d_model=2,
        num_heads=1,
        d_ff=2,
        dropout=0.0,
        norm="rmsnorm",
        ffn="swiglu",
        rope=rope,
    )
    x = Tensor([[[0.3, -0.2], [0.6, 0.1]]])

    assert gradcheck(
        lambda value: block(value),
        x,
        parameters=block.named_parameters(),
        atol=1e-5,
        rtol=1e-4,
    )
