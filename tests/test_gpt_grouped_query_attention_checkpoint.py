import numpy as np
import pytest

from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint
from nn import GroupedQueryAttention, MultiHeadAttention
from nn.transformer import GPT


def _config(**overrides):
    config = {
        "vocab_size": 19,
        "context_len": 6,
        "d_model": 8,
        "num_heads": 4,
        "d_ff": 16,
        "num_layers": 2,
        "dropout": 0.0,
        "norm": "rmsnorm",
        "pos_encoding": "rope",
        "ffn": "swiglu",
    }
    config.update(overrides)
    return config


def _assert_same_model_outputs(left, right):
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    np.testing.assert_array_equal(left(tokens).data, right(tokens).data)
    left_logits, left_cache = left.infer(tokens)
    right_logits, right_cache = right.infer(tokens)
    np.testing.assert_array_equal(left_logits, right_logits)
    for left_entry, right_entry in zip(left_cache, right_cache):
        np.testing.assert_array_equal(left_entry["k"], right_entry["k"])
        np.testing.assert_array_equal(left_entry["v"], right_entry["v"])


def test_gqa_config_is_self_describing_and_round_trips_model_state():
    np.random.seed(1010)
    source = GPT(**_config(num_kv_heads=2))
    config = source.config()

    assert config["num_kv_heads"] == 2
    recreated = GPT(**config)
    assert recreated.num_kv_heads == 2
    assert all(isinstance(block.attn, GroupedQueryAttention) for block in recreated.blocks)

    recreated.load_state_dict(source.state_dict())
    _assert_same_model_outputs(source, recreated)


def test_legacy_mha_config_shape_is_unchanged_and_recreates_existing_attention():
    model = GPT(**_config())
    config = model.config()

    assert "num_kv_heads" not in config
    assert set(config) == {
        "vocab_size",
        "context_len",
        "d_model",
        "num_heads",
        "d_ff",
        "num_layers",
        "dropout",
        "lora_rank",
        "lora_alpha",
        "norm",
        "pos_encoding",
        "ffn",
    }
    recreated = GPT(**config)
    assert recreated.num_kv_heads == recreated.num_heads
    assert all(isinstance(block.attn, MultiHeadAttention) for block in recreated.blocks)


def test_pickle_checkpoint_metadata_recreates_gqa_model(tmp_path):
    np.random.seed(2020)
    source = GPT(**_config(num_kv_heads=2))
    path = tmp_path / "gqa.pkl"
    save_checkpoint(
        path,
        source,
        step=7,
        metadata={"model_config": source.config()},
    )

    checkpoint = read_checkpoint(path)
    assert checkpoint["metadata"]["model_config"]["num_kv_heads"] == 2
    restored = GPT(**checkpoint["metadata"]["model_config"])
    assert restore_checkpoint(checkpoint, restored) == 7
    _assert_same_model_outputs(source, restored)


def test_safe_checkpoint_metadata_recreates_gqa_model(tmp_path):
    np.random.seed(3030)
    source = GPT(**_config(num_kv_heads=1))
    path = tmp_path / "gqa-safe.npz"
    save_safe_checkpoint(
        path,
        source,
        step=9,
        metadata={"model_config": source.config()},
    )

    checkpoint = read_safe_checkpoint(path)
    assert checkpoint["metadata"]["model_config"]["num_kv_heads"] == 1
    restored = GPT(**checkpoint["metadata"]["model_config"])
    assert restore_checkpoint(checkpoint, restored) == 9
    assert restored.num_kv_heads == 1
    _assert_same_model_outputs(source, restored)


def test_gqa_checkpoint_rejects_reconstruction_with_wrong_kv_head_layout_transactionally(tmp_path):
    np.random.seed(4040)
    source = GPT(**_config(num_kv_heads=2))
    path = tmp_path / "gqa.pkl"
    save_checkpoint(
        path,
        source,
        metadata={"model_config": source.config()},
    )
    checkpoint = read_checkpoint(path)

    target = GPT(**_config())
    before = target.state_dict()
    with pytest.raises(ValueError, match="shape mismatch"):
        restore_checkpoint(checkpoint, target)
    after = target.state_dict()
    assert tuple(after) == tuple(before)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_explicit_full_kv_heads_canonicalize_to_legacy_checkpoint_config():
    model = GPT(**_config(num_kv_heads=4))
    assert model.num_kv_heads == 4
    assert "num_kv_heads" not in model.config()
