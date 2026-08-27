"""Regression tests for streaming-checkpoint preflight validation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import save_checkpoint
from engine.safe_checkpoint import save_safe_checkpoint
from nn.transformer import GPT
from streaming_cli import load_streaming_checkpoint
from tokenizer import CharTokenizer


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _write_checkpoint(tmp_path, *, checkpoint_format, model, tokenizer, name):
    path = tmp_path / (
        f"{name}.pkl" if checkpoint_format == "pickle" else f"{name}.npz"
    )
    metadata = {
        "model_config": model.config(),
        "tokenizer": tokenizer.state_dict(),
    }
    if checkpoint_format == "pickle":
        save_checkpoint(path, model, metadata=metadata)
    else:
        save_safe_checkpoint(path, model, metadata=metadata)
    return path


def _rope_model(vocab_size):
    return GPT(
        vocab_size=vocab_size,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        norm="rmsnorm",
        pos_encoding="rope",
        ffn="swiglu",
    )


def _learned_model(vocab_size):
    return GPT(
        vocab_size=vocab_size,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        norm="layernorm",
        pos_encoding="learned",
        ffn="gelu",
    )


@pytest.mark.parametrize("checkpoint_format", ["pickle", "safe"])
def test_learned_position_rejection_does_not_restore_checkpoint_rng(
    tmp_path, checkpoint_format
):
    tokenizer = CharTokenizer.train("abcde ")
    model = _learned_model(tokenizer.vocab_size)

    np.random.seed(111)
    path = _write_checkpoint(
        tmp_path,
        checkpoint_format=checkpoint_format,
        model=model,
        tokenizer=tokenizer,
        name="learned",
    )

    np.random.seed(999)
    before = np.random.get_state()
    with pytest.raises(ValueError, match="requires a RoPE checkpoint"):
        load_streaming_checkpoint(path, checkpoint_format=checkpoint_format)
    _assert_rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize("checkpoint_format", ["pickle", "safe"])
def test_vocab_mismatch_rejection_does_not_restore_checkpoint_rng(
    tmp_path, checkpoint_format
):
    model_tokenizer = CharTokenizer.train("abcd")
    metadata_tokenizer = CharTokenizer.train("abcde")
    model = _rope_model(model_tokenizer.vocab_size)

    np.random.seed(222)
    path = _write_checkpoint(
        tmp_path,
        checkpoint_format=checkpoint_format,
        model=model,
        tokenizer=metadata_tokenizer,
        name="vocab-mismatch",
    )

    np.random.seed(888)
    before = np.random.get_state()
    with pytest.raises(ValueError, match="vocabulary sizes differ"):
        load_streaming_checkpoint(path, checkpoint_format=checkpoint_format)
    _assert_rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize("checkpoint_format", ["pickle", "safe"])
def test_valid_streaming_checkpoint_keeps_existing_restore_semantics(
    tmp_path, checkpoint_format
):
    tokenizer = CharTokenizer.train("abcde ")
    model = _rope_model(tokenizer.vocab_size)

    np.random.seed(333)
    expected_rng = np.random.get_state()
    path = _write_checkpoint(
        tmp_path,
        checkpoint_format=checkpoint_format,
        model=model,
        tokenizer=tokenizer,
        name="valid",
    )

    np.random.seed(777)
    restored_model, restored_tokenizer = load_streaming_checkpoint(
        path, checkpoint_format=checkpoint_format
    )

    assert restored_model.rope is not None
    assert restored_model.training is False
    assert restored_tokenizer.state_dict() == tokenizer.state_dict()
    _assert_rng_state_equal(np.random.get_state(), expected_rng)
