"""Tests for training and inference extensions."""

import os
import sys
import json
from argparse import Namespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.tensor import Tensor
from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from engine.optim import Adam
from engine.scheduler import WarmupCosineScheduler
from nn.transformer import GPT, _sample
from tokenizer import BPETokenizer, CharTokenizer
from benchmark import run_benchmark
from train import _append_jsonl, _prompt_array, _validate_args, clip_grad_norm_


def make_model(**overrides):
    config = {
        "vocab_size": 12,
        "context_len": 8,
        "d_model": 8,
        "num_heads": 2,
        "d_ff": 16,
        "num_layers": 1,
    }
    config.update(overrides)
    return GPT(**config)


def test_warmup_cosine_scheduler_curve():
    model = make_model()
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=5, warmup_steps=2, min_lr=0.01)
    rates = [scheduler.step(step) for step in range(5)]
    np.testing.assert_allclose(rates, [0.05, 0.1, 0.1, 0.055, 0.01])


def test_checkpoint_roundtrip(tmp_path):
    model = make_model()
    optimizer = Adam(model.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=5, warmup_steps=1)
    scheduler.step(2)
    path = tmp_path / "model.ckpt"
    save_checkpoint(path, model, optimizer, scheduler, step=3, metadata={"name": "test"})

    restored = make_model()
    restored_optimizer = Adam(restored.parameters(), lr=0.01)
    restored_scheduler = WarmupCosineScheduler(restored_optimizer, total_steps=2)
    state = read_checkpoint(path)
    assert restore_checkpoint(state, restored, restored_optimizer, restored_scheduler) == 3
    assert state["metadata"] == {"name": "test"}
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value, restored.state_dict()[name])
    assert restored_optimizer.t == optimizer.t
    assert restored_scheduler.last_step == scheduler.last_step


def test_bpe_tokenizer_roundtrip_and_compression():
    text = "banana bandana banana bandana\n"
    tokenizer = BPETokenizer.train(text, num_merges=10)
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text
    assert len(encoded) < len(text)


def test_cached_inference_matches_full_inference():
    np.random.seed(1)
    model = make_model()
    tokens = np.array([[1, 2, 3, 4]])
    full, _ = model.infer(tokens)
    _, cache = model.infer(tokens[:, :3])
    incremental, cache = model.infer(tokens[:, 3:], cache)
    np.testing.assert_allclose(full[:, -1], incremental[:, -1], atol=1e-10)
    assert cache[0]["k"].shape == (1, 2, 4, 4)


def test_generation_strategies_and_filters():
    np.random.seed(2)
    model = make_model(context_len=4)
    prompt = np.array([[1, 2, 3]])
    cached = model.generate(prompt, 5, strategy="greedy", use_cache=True)
    uncached = model.generate(prompt, 5, strategy="greedy", use_cache=False)
    np.testing.assert_array_equal(cached, uncached)
    assert model.generate(prompt, 2, strategy="beam", beam_width=2).shape == (1, 5)
    assert _sample(np.array([10.0, 1.0, 0.0]), top_k=1) == 0
    assert _sample(np.array([10.0, 1.0, 0.0]), top_p=0.5) == 0


def test_top_p_keeps_threshold_crossing_token(monkeypatch):
    seen = {}

    def choose(_, p):
        seen["probabilities"] = p
        return 0

    monkeypatch.setattr(np.random, "choice", choose)
    _sample(np.log(np.array([0.4, 0.35, 0.25])), top_p=0.6)
    assert np.count_nonzero(seen["probabilities"]) == 2


def test_lora_freezes_backbone_and_receives_gradients():
    np.random.seed(3)
    model = make_model(lora_rank=2, lora_alpha=4)
    trainable = list(model.named_parameters())
    assert trainable
    assert all("lora_" in name for name, _ in trainable)

    tokens = np.array([[1, 2, 3]])
    ops.sum(model(tokens)).backward()
    b_gradients = [
        parameter.grad
        for name, parameter in trainable
        if name.endswith("lora_B")
    ]
    assert b_gradients
    assert any(np.any(gradient != 0) for gradient in b_gradients)


def test_prompt_array_uses_custom_prompt_and_crops_context():
    tokenizer = CharTokenizer.train("hello world")
    args = Namespace(prompt="hello", prompt_file=None)
    prompt = _prompt_array(args, tokenizer, "ignored", context_len=3)
    assert prompt.shape == (1, 3)
    assert tokenizer.decode(prompt[0]) == "llo"


def test_prompt_array_rejects_empty_prompt():
    tokenizer = CharTokenizer.train("abc")
    args = Namespace(prompt="", prompt_file=None)
    with pytest.raises(ValueError, match="must not be empty"):
        _prompt_array(args, tokenizer, "abc", context_len=3)


def test_arg_validation_accepts_good_defaults_and_rejects_conflicts():
    args = _valid_args()
    _validate_args(args)

    args = _valid_args(prompt="x", prompt_file="prompt.txt")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)

    args = _valid_args(eval_only=True, generate_only=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_args(args)


def test_clip_grad_norm_zero_disables_scaling():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad[:] = np.array([3.0, 4.0])
    total = clip_grad_norm_([tensor], max_norm=0.0)
    assert total == 5.0
    np.testing.assert_array_equal(tensor.grad, np.array([3.0, 4.0]))


def test_append_jsonl_writes_sorted_json_record(tmp_path):
    path = tmp_path / "metrics" / "train.jsonl"
    _append_jsonl(path, {"step": 1, "train_loss": 2.0})
    _append_jsonl(path, {"step": 2, "train_loss": 1.5})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"step": 1, "train_loss": 2.0},
        {"step": 2, "train_loss": 1.5},
    ]


def test_benchmark_returns_positive_metrics():
    args = Namespace(
        vocab=16,
        ctx=4,
        d=8,
        heads=2,
        layers=1,
        batch=1,
        steps=1,
        generate=1,
        seed=0,
    )
    metrics = run_benchmark(args)
    assert metrics["infer_tokens_per_sec"] > 0
    assert metrics["generate_cached_tokens_per_sec"] > 0
    assert metrics["generate_uncached_tokens_per_sec"] > 0
    assert metrics["cache_speedup"] > 0


def test_benchmark_rejects_invalid_head_shape():
    args = Namespace(
        vocab=16,
        ctx=4,
        d=7,
        heads=2,
        layers=1,
        batch=1,
        steps=1,
        generate=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="divisible"):
        run_benchmark(args)


def _valid_args(**overrides):
    defaults = {
        "iters": 1,
        "batch": 1,
        "ctx": 4,
        "d": 8,
        "heads": 2,
        "layers": 1,
        "eval_interval": 1,
        "eval_iters": 1,
        "warmup_iters": 0,
        "save_every": 0,
        "val_frac": 0.1,
        "lr": 1e-3,
        "min_lr": 0.0,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "bpe_merges": 0,
        "lora_rank": 0,
        "lora_alpha": 1.0,
        "sample": 0,
        "temperature": 1.0,
        "beam_width": 1,
        "top_k": None,
        "top_p": None,
        "prompt": None,
        "prompt_file": None,
        "eval_only": False,
        "generate_only": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)
