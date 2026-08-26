"""Tests for the checkpoint-backed streaming generation command."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import save_checkpoint
from engine.safe_checkpoint import save_safe_checkpoint
from nn.transformer import GPT
import streaming_cli
from streaming_cli import load_streaming_checkpoint, main
from tokenizer import CharTokenizer


def _checkpoint(
    tmp_path,
    pos_encoding="rope",
    include_metadata=True,
    checkpoint_format="pickle",
):
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
    if checkpoint_format == "pickle":
        path = tmp_path / f"{pos_encoding}.pkl"
        save_checkpoint(path, model, metadata=metadata)
    elif checkpoint_format == "safe":
        path = tmp_path / f"{pos_encoding}.npz"
        save_safe_checkpoint(path, model, metadata=metadata)
    else:
        raise AssertionError(f"unsupported test checkpoint format: {checkpoint_format}")
    return path, tokenizer


@pytest.mark.parametrize("checkpoint_format", ["pickle", "safe"])
def test_load_streaming_checkpoint_restores_rope_model(
    tmp_path,
    checkpoint_format,
):
    path, tokenizer = _checkpoint(tmp_path, checkpoint_format=checkpoint_format)

    model, restored_tokenizer = load_streaming_checkpoint(
        path,
        checkpoint_format=checkpoint_format,
    )

    assert model.rope is not None
    assert model.training is False
    assert restored_tokenizer.state_dict() == tokenizer.state_dict()


def test_load_streaming_checkpoint_defaults_to_trusted_pickle(tmp_path):
    path, tokenizer = _checkpoint(tmp_path)

    model, restored_tokenizer = load_streaming_checkpoint(path)

    assert model.rope is not None
    assert restored_tokenizer.state_dict() == tokenizer.state_dict()


def test_load_streaming_checkpoint_rejects_unknown_format(tmp_path):
    path, _ = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="checkpoint_format"):
        load_streaming_checkpoint(path, checkpoint_format="unknown")
    with pytest.raises(TypeError, match="checkpoint_format"):
        load_streaming_checkpoint(path, checkpoint_format=None)


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


def test_cli_safe_checkpoint_zero_token_round_trip(monkeypatch, tmp_path, capsys):
    path, _ = _checkpoint(tmp_path, checkpoint_format="safe")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            str(path),
            "--checkpoint-format",
            "safe",
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


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--temperature", "nan", "temperature.*finite"),
        ("--temperature", "inf", "temperature.*finite"),
        ("--temperature", "-inf", "temperature.*finite"),
        ("--top-p", "nan", "top_p.*finite"),
        ("--top-p", "inf", "top_p.*finite"),
        ("--seed", "-1", "seed.*non-negative"),
        ("--seed", str(2**32), "seed.*at most"),
    ],
)
def test_cli_invalid_numeric_options_fail_before_checkpoint_read(
    monkeypatch,
    flag,
    value,
    message,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            "must-not-be-read.pkl",
            "--prompt",
            "abc",
            flag,
            value,
        ],
    )

    def unexpected_checkpoint_read(*_args, **_kwargs):
        raise AssertionError("invalid CLI options must fail before checkpoint I/O")

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        unexpected_checkpoint_read,
    )

    with pytest.raises(ValueError, match=message):
        streaming_cli.main()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokens", True),
        ("tokens", 1.5),
        ("top_k", True),
        ("top_k", 1.5),
        ("seed", True),
        ("seed", 1.5),
    ],
)
def test_programmatic_cli_validation_rejects_non_integer_fields(field, value):
    args = streaming_cli.parse_args
    values = {
        "tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "seed": 7,
        "strategy": "sample",
    }
    values[field] = value
    namespace = type("Args", (), values)()

    with pytest.raises(TypeError):
        streaming_cli._validate_args(namespace)
