import numpy as np
import pytest

import engine.ops as ops
from nn import GroupedQueryAttention, MultiHeadAttention
from nn.transformer import GPT, TransformerBlock


def _gpt(**overrides):
    config = {
        "vocab_size": 17,
        "context_len": 6,
        "d_model": 8,
        "num_heads": 4,
        "d_ff": 16,
        "num_layers": 2,
        "dropout": 0.0,
    }
    config.update(overrides)
    return GPT(**config)


def test_gpt_uses_grouped_attention_only_when_kv_heads_are_reduced():
    grouped = _gpt(num_kv_heads=2)
    mqa = _gpt(num_kv_heads=1)
    legacy = _gpt()

    assert grouped.num_kv_heads == 2
    assert mqa.num_kv_heads == 1
    assert legacy.num_kv_heads == legacy.num_heads == 4
    assert all(isinstance(block.attn, GroupedQueryAttention) for block in grouped.blocks)
    assert all(isinstance(block.attn, GroupedQueryAttention) for block in mqa.blocks)
    assert all(isinstance(block.attn, MultiHeadAttention) for block in legacy.blocks)
    assert all(block.attn.W_k.weight.shape == (4, 8) for block in grouped.blocks)
    assert all(block.attn.W_v.weight.shape == (4, 8) for block in grouped.blocks)
    assert all(block.attn.W_k.weight.shape == (2, 8) for block in mqa.blocks)
    assert all(block.attn.W_v.weight.shape == (2, 8) for block in mqa.blocks)


def test_default_and_explicit_full_kv_head_models_keep_exact_legacy_trajectory():
    np.random.seed(1234)
    default = _gpt()
    default_state = default.state_dict()

    np.random.seed(1234)
    explicit = _gpt(num_kv_heads=4)
    explicit_state = explicit.state_dict()

    assert default.config() == explicit.config()
    assert "num_kv_heads" not in default.config()
    assert tuple(default_state) == tuple(explicit_state)
    for name in default_state:
        np.testing.assert_array_equal(default_state[name], explicit_state[name])

    tokens = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    np.testing.assert_array_equal(default(tokens).data, explicit(tokens).data)

    default_logits, default_cache = default.infer(tokens)
    explicit_logits, explicit_cache = explicit.infer(tokens)
    np.testing.assert_array_equal(default_logits, explicit_logits)
    for left, right in zip(default_cache, explicit_cache):
        np.testing.assert_array_equal(left["k"], right["k"])
        np.testing.assert_array_equal(left["v"], right["v"])


def test_gqa_gpt_forward_backward_updates_compact_kv_projection_gradients():
    np.random.seed(222)
    model = _gpt(num_kv_heads=2)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5]], dtype=np.int64)

    loss = ops.cross_entropy(model(tokens), targets)
    loss.backward()

    assert np.isfinite(float(loss.data))
    for block in model.blocks:
        assert block.attn.W_k.weight.grad.shape == (4, 8)
        assert block.attn.W_v.weight.grad.shape == (4, 8)
        assert np.isfinite(block.attn.W_k.weight.grad).all()
        assert np.isfinite(block.attn.W_v.weight.grad).all()
        assert np.any(block.attn.W_k.weight.grad != 0.0)
        assert np.any(block.attn.W_v.weight.grad != 0.0)


def test_gqa_gpt_supports_rope_and_gradient_checkpointing_with_same_result():
    config = {
        "num_kv_heads": 2,
        "norm": "rmsnorm",
        "pos_encoding": "rope",
        "ffn": "swiglu",
    }
    np.random.seed(333)
    plain = _gpt(**config)
    state = plain.state_dict()
    checkpointed = _gpt(**config, grad_checkpoint=True)
    checkpointed.load_state_dict(state)

    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    targets = np.array([[2, 3, 4]], dtype=np.int64)
    plain_loss = ops.cross_entropy(plain(tokens), targets)
    checkpointed_loss = ops.cross_entropy(checkpointed(tokens), targets)
    np.testing.assert_array_equal(plain_loss.data, checkpointed_loss.data)

    plain_loss.backward()
    checkpointed_loss.backward()
    plain_grads = dict(plain.named_parameters())
    checkpointed_grads = dict(checkpointed.named_parameters())
    assert tuple(plain_grads) == tuple(checkpointed_grads)
    for name in plain_grads:
        np.testing.assert_allclose(
            plain_grads[name].grad,
            checkpointed_grads[name].grad,
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (np.bool_(False), TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (3, ValueError),
        (8, ValueError),
    ],
)
def test_gpt_validates_num_kv_heads_before_parameter_initialization(value, error):
    np.random.seed(444)
    before = np.random.get_state()
    with pytest.raises(error):
        _gpt(num_kv_heads=value)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_numpy_integer_kv_heads_are_accepted_and_repr_records_grouping():
    model = _gpt(num_kv_heads=np.int64(2))
    assert model.num_kv_heads == 2
    assert "kv_heads=2" in repr(model)


def test_transformer_block_uses_same_attention_selection_contract():
    grouped = TransformerBlock(8, 4, 16, num_kv_heads=2)
    legacy = TransformerBlock(8, 4, 16)
    assert isinstance(grouped.attn, GroupedQueryAttention)
    assert isinstance(legacy.attn, MultiHeadAttention)
    assert grouped.num_kv_heads == 2
    assert legacy.num_kv_heads == 4
