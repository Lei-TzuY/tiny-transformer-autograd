"""Tests for the checkpoint-backed streaming generation command."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import save_checkpoint
from nn.transformer import GPT
from streaming_cli import load_streaming_checkpoint, main
from tokenizer import CharTokenizer


def _checkpoint(tmp_path, pos_encoding="rope", include_metadata=True):
    tokenizer = CharTokenizer.train("abcde ")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        context_len=8,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        norm="rmsnorm" if pos_encoding == "rope" else "layernorm",
        pos_encoding=pos_encoding,
        ffn="swiglu" if pos_encoding == "rope" else "gelu",
    )
    metadata = (
        {"model_config": model.config(), "tokenizer": tokenizer.state_dict()}
        if include_metadata
        else None
    )
    path = tmp_path / f"{pos_encoding}.pkl"
    save_checkpoint(path, model, metadata=metadata)
    return path, tokenizer


def test_load_streaming_checkpoint_restores_rope_model(tmp_path):
    path, tokenizer = _checkpoint(tmp_path)

    model, restored_tokenizer = load_streaming_checkpoint(path)

    assert model.rope is not None
    assert model.training is False
    assert restored_tokenizer.state_dict() == tokenizer.state_dict()


def test_load_streaming_checkpoint_rejects_learned_positions(tmp_path):
    path, _ = _checkpoint(tmp_path, pos_encoding="learned")

    with pytest.raises(ValueError, match="requires a RoPE checkpoint"):
        load_streaming_checkpoint(path)


def test_load_streaming_checkpoint_requires_metadata(tmp_path):
    path, _ = _checkpoint(tmp_path, include_metadata=False)

    with pytest.raises(ValueError, match="requires checkpoint metadata"):
        load_streaming_checkpoint(path)


def test_cli_zero_token_greedy_round_trips_prompt(monkeypatch, tmp_path, capsys):
    path, _ = _checkpoint(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            str(path),
            "--prompt",
            "abc",
            "--tokens",
            "0",
            "--strategy",
            "greedy",
        ],
    )

    main()

    assert capsys.readouterr().out.strip() == "abc"


def test_cli_reads_prompt_file(monkeypatch, tmp_path, capsys):
    checkpoint, _ = _checkpoint(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("de a", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            str(checkpoint),
            "--prompt-file",
            str(prompt),
            "--tokens",
            "0",
        ],
    )

    main()

    assert capsys.readouterr().out.strip() == "de a"


def test_cli_reports_out_of_vocabulary_prompt(monkeypatch, tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "z",
            "--tokens",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="not present in the tokenizer vocabulary"):
        main()
