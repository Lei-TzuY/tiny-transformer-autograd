import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.checkpoint import CHECKPOINT_VERSION, restore_checkpoint
from engine.optim import AdamW
from engine.scheduler import WarmupCosineScheduler
from nn import (
    GPT,
    convert_gpt_checkpoint_kv_heads,
    convert_gpt_kv_heads,
)


def _model(*, kv_heads=None, lora_rank=0):
    np.random.seed(17)
    kwargs = {}
    if kv_heads is not None:
        kwargs["num_kv_heads"] = kv_heads
    return GPT(
        vocab_size=19,
        context_len=5,
        d_model=8,
        num_heads=4,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        lora_rank=lora_rank,
        lora_alpha=3.0,
        **kwargs,
    )


def _train_state(model, *, step=7):
    optimizer = AdamW(
        model.parameters(),
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.03,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=30,
        warmup_steps=3,
        min_lr=0.001,
    )
    scheduler.step(5)
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = np.full_like(parameter.data, index / 100.0)
    optimizer.step()
    return optimizer, scheduler, {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_type": "AdamW",
        "scheduler": scheduler.state_dict(),
        "rng_state": np.random.get_state(),
        "step": step,
        "metadata": {
            "model_config": model.config(),
            "tokenizer": {"kind": "fixture"},
            "user_note": "preserve me",
        },
    }


def _assert_state_equal(left, right):
    assert set(left) == set(right)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_mha_checkpoint_to_gqa_resets_optimizer_moments_and_resumes():
    source = _model()
    source_optimizer, source_scheduler, checkpoint = _train_state(source)
    expected_model = convert_gpt_kv_heads(source, 2)

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    config = converted["metadata"]["model_config"]
    assert config["num_kv_heads"] == 2
    target = GPT(**config)
    target.load_state_dict(converted["model"], strict=True)
    _assert_state_equal(target.state_dict(), expected_model.state_dict())

    saved_optimizer = converted["optimizer"]
    assert converted["optimizer_type"] == "AdamW"
    assert saved_optimizer["lr"] == source_optimizer.lr
    assert saved_optimizer["betas"] == (
        source_optimizer.beta1,
        source_optimizer.beta2,
    )
    assert saved_optimizer["eps"] == source_optimizer.eps
    assert saved_optimizer["weight_decay"] == source_optimizer.weight_decay
    assert saved_optimizer["t"] == 0
    assert saved_optimizer["steps"] == [0] * len(target.parameters())
    assert [value.shape for value in saved_optimizer["m"]] == [
        parameter.data.shape for parameter in target.parameters()
    ]
    assert [value.shape for value in saved_optimizer["v"]] == [
        parameter.data.shape for parameter in target.parameters()
    ]
    assert all(np.count_nonzero(value) == 0 for value in saved_optimizer["m"])
    assert all(np.count_nonzero(value) == 0 for value in saved_optimizer["v"])

    assert converted["scheduler"] == source_scheduler.state_dict()
    assert converted["step"] == checkpoint["step"]
    assert converted["metadata"]["tokenizer"] == {"kind": "fixture"}
    assert converted["metadata"]["user_note"] == "preserve me"
    assert converted["metadata"]["_tiny_transformer_migrations"][-1] == {
        "kind": "gpt_kv_heads",
        "source_num_kv_heads": 4,
        "target_num_kv_heads": 2,
        "optimizer_state": "reset",
    }

    resumed_optimizer = AdamW(target.parameters(), lr=1e-4)
    resumed_scheduler = WarmupCosineScheduler(
        resumed_optimizer, total_steps=1, warmup_steps=0
    )
    restored_step = restore_checkpoint(
        converted,
        target,
        resumed_optimizer,
        resumed_scheduler,
    )
    assert restored_step == checkpoint["step"]
    assert resumed_optimizer.t == 0
    assert resumed_optimizer._steps == [0] * len(target.parameters())
    assert resumed_scheduler.state_dict() == source_scheduler.state_dict()

    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    targets = np.array([[2, 3, 4]], dtype=np.int64)
    resumed_optimizer.zero_grad(set_to_none=True)
    loss = ops.cross_entropy(target(tokens), targets)
    loss.backward()
    resumed_optimizer.step()
    assert resumed_optimizer.t == 1
    assert any(step == 1 for step in resumed_optimizer._steps)


def test_gqa_checkpoint_expands_to_mha_model_state():
    source = _model(kv_heads=2)
    _, _, checkpoint = _train_state(source, step=9)
    expected = convert_gpt_kv_heads(source, 4)

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 4)

    assert "num_kv_heads" not in converted["metadata"]["model_config"]
    target = GPT(**converted["metadata"]["model_config"])
    target.load_state_dict(converted["model"], strict=True)
    _assert_state_equal(target.state_dict(), expected.state_dict())
    assert converted["metadata"]["_tiny_transformer_migrations"][-1][
        "source_num_kv_heads"
    ] == 2
    assert converted["metadata"]["_tiny_transformer_migrations"][-1][
        "target_num_kv_heads"
    ] == 4


def test_same_head_checkpoint_clone_preserves_nonzero_optimizer_state_exactly():
    source = _model(kv_heads=2)
    _, _, checkpoint = _train_state(source)
    original_m = [value.copy() for value in checkpoint["optimizer"]["m"]]
    original_v = [value.copy() for value in checkpoint["optimizer"]["v"]]
    assert any(np.count_nonzero(value) for value in original_m)

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    assert "_tiny_transformer_migrations" not in converted["metadata"]
    assert converted["optimizer"]["t"] == checkpoint["optimizer"]["t"]
    assert converted["optimizer"]["steps"] == checkpoint["optimizer"]["steps"]
    for actual, expected in zip(converted["optimizer"]["m"], original_m):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    for actual, expected in zip(converted["optimizer"]["v"], original_v):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)

    first_name = next(iter(converted["model"]))
    converted["model"][first_name].flat[0] += 1.0
    assert not np.array_equal(converted["model"][first_name], checkpoint["model"][first_name])


def test_checkpoint_without_optimizer_keeps_training_metadata_and_marks_absence():
    source = _model()
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "model": source.state_dict(),
        "optimizer": None,
        "optimizer_type": None,
        "scheduler": None,
        "rng_state": np.random.get_state(),
        "step": 11,
        "metadata": {"model_config": source.config(), "tag": "eval-only"},
    }

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 1)

    assert converted["optimizer"] is None
    assert converted["optimizer_type"] is None
    assert converted["scheduler"] is None
    assert converted["step"] == 11
    assert converted["metadata"]["tag"] == "eval-only"
    assert converted["metadata"]["_tiny_transformer_migrations"][-1][
        "optimizer_state"
    ] == "absent"


def test_lora_checkpoint_reset_state_tracks_converted_trainable_shapes():
    source = _model(lora_rank=2)
    _, _, checkpoint = _train_state(source)

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)
    target = GPT(**converted["metadata"]["model_config"])
    target.load_state_dict(converted["model"], strict=True)

    assert converted["metadata"]["model_config"]["lora_rank"] == 2
    assert converted["metadata"]["model_config"]["lora_alpha"] == 3.0
    assert [value.shape for value in converted["optimizer"]["m"]] == [
        parameter.data.shape for parameter in target.parameters()
    ]
    assert [value.shape for value in converted["optimizer"]["v"]] == [
        parameter.data.shape for parameter in target.parameters()
    ]
    assert target.blocks[0].attn.W_k.lora_B.data.shape[0] == 4
    assert target.blocks[0].attn.W_v.lora_B.data.shape[0] == 4


def test_conversion_preserves_checkpoint_rng_and_global_rng_state():
    source = _model()
    _, _, checkpoint = _train_state(source)
    saved_checkpoint_rng = checkpoint["rng_state"]

    np.random.seed(24681357)
    caller_rng = np.random.get_state()
    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    _assert_rng_equal(np.random.get_state(), caller_rng)
    _assert_rng_equal(converted["rng_state"], saved_checkpoint_rng)
