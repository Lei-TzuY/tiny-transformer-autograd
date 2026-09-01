import numpy as np

import engine.ops as ops
from nn.transformer import GPT


def _model():
    np.random.seed(6161)
    return GPT(
        vocab_size=17,
        context_len=5,
        d_model=8,
        num_heads=4,
        num_kv_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        lora_rank=2,
        lora_alpha=4.0,
    )


def test_lora_uses_compact_adapter_shapes_for_gqa_kv_projections():
    model = _model()
    attention = model.blocks[0].attn

    assert attention.W_q.lora_B.shape == (8, 2)
    assert attention.W_k.lora_B.shape == (4, 2)
    assert attention.W_v.lora_B.shape == (4, 2)
    assert attention.out_proj.lora_B.shape == (8, 2)
    assert model.config()["num_kv_heads"] == 2
    assert model.config()["lora_rank"] == 2
    assert model.config()["lora_alpha"] == 4.0


def test_lora_gqa_backward_updates_compact_trainable_adapters():
    model = _model()
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    targets = np.array([[2, 3, 4]], dtype=np.int64)

    loss = ops.cross_entropy(model(tokens), targets)
    loss.backward()

    attention = model.blocks[0].attn
    assert attention.W_k.weight.requires_grad is False
    assert attention.W_v.weight.requires_grad is False
    assert attention.W_k.weight.grad is None
    assert attention.W_v.weight.grad is None
    assert attention.W_k.lora_B.requires_grad is True
    assert attention.W_v.lora_B.requires_grad is True
    assert attention.W_k.lora_B.grad.shape == (4, 2)
    assert attention.W_v.lora_B.grad.shape == (4, 2)
    assert np.isfinite(attention.W_k.lora_B.grad).all()
    assert np.isfinite(attention.W_v.lora_B.grad).all()
