"""Regression tests for token-by-token ``tiny-stream`` output."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streaming_cli
from tokenizer import BPETokenizer, CharTokenizer


class _ModelStub:
    def __init__(self, vocab_size, context_len=4):
        self.vocab_size = vocab_size
        self.context_len = context_len


class _RecordingStdout:
    def __init__(self):
        self.parts = []
        self.flush_count = 0

    @property
    def text(self):
        return "".join(self.parts)

    def write(self, value):
        self.parts.append(value)
        return len(value)

    def flush(self):
        self.flush_count += 1


def _set_cli(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            "fake.pkl",
            "--prompt",
            "ab",
            "--tokens",
            "2",
            "--strategy",
            "greedy",
            *extra,
        ],
    )


def test_incremental_output_flushes_each_piece_before_requesting_next(monkeypatch):
    tokenizer = CharTokenizer.train("abc ")
    model = _ModelStub(tokenizer.vocab_size)
    output = _RecordingStdout()
    seen = {}

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        lambda *_args, **_kwargs: (model, tokenizer),
    )

    def fail_buffered(*_args, **_kwargs):
        raise AssertionError("incremental output must not call stream_generate")

    monkeypatch.setattr(streaming_cli, "stream_generate", fail_buffered)

    def fake_iter(received_model, visible_prompt, **kwargs):
        seen["model"] = received_model
        seen["prompt"] = visible_prompt.copy()
        seen["kwargs"] = kwargs.copy()

        def steps():
            assert output.text == "ab"
            assert output.flush_count >= 1
            yield np.array([tokenizer.stoi["c"]], dtype=np.int64)
            assert output.text == "abc"
            assert output.flush_count >= 2
            yield np.array([tokenizer.stoi[" "]], dtype=np.int64)

        return steps()

    monkeypatch.setattr(streaming_cli, "stream_generate_iter", fake_iter)
    monkeypatch.setattr(sys, "stdout", output)
    _set_cli(
        monkeypatch,
        "--incremental-output",
        "--stop-token-id",
        str(tokenizer.stoi[" "]),
    )

    streaming_cli.main()

    assert output.text == "abc \n"
    assert output.flush_count >= 4
    assert seen["model"] is model
    np.testing.assert_array_equal(
        seen["prompt"],
        np.array([[tokenizer.stoi["a"], tokenizer.stoi["b"]]], dtype=np.int64),
    )
    assert seen["kwargs"] == {
        "max_new_tokens": 2,
        "temperature": 0.8,
        "top_k": None,
        "top_p": None,
        "strategy": "greedy",
        "stop_token_id": tokenizer.stoi[" "],
    }


def test_default_output_keeps_buffered_generation_path(monkeypatch, capsys):
    tokenizer = CharTokenizer.train("abc ")
    model = _ModelStub(tokenizer.vocab_size)
    stop_id = tokenizer.stoi["c"]
    seen = {}

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        lambda *_args, **_kwargs: (model, tokenizer),
    )

    def fail_iterator(*_args, **_kwargs):
        raise AssertionError("default output must preserve the buffered path")

    monkeypatch.setattr(streaming_cli, "stream_generate_iter", fail_iterator)

    def fake_generate(received_model, visible_prompt, **kwargs):
        seen["model"] = received_model
        seen["prompt"] = visible_prompt.copy()
        seen["kwargs"] = kwargs.copy()
        return np.array(
            [[tokenizer.stoi["a"], tokenizer.stoi["b"], stop_id]],
            dtype=np.int64,
        )

    monkeypatch.setattr(streaming_cli, "stream_generate", fake_generate)
    _set_cli(monkeypatch, "--stop-token-id", str(stop_id))

    streaming_cli.main()

    assert capsys.readouterr().out == "abc\n"
    assert seen["model"] is model
    assert seen["kwargs"]["stop_token_id"] == stop_id


def test_incremental_zero_tokens_prints_visible_prompt_once(monkeypatch):
    tokenizer = CharTokenizer.train("abc ")
    model = _ModelStub(tokenizer.vocab_size)
    output = _RecordingStdout()
    calls = []

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        lambda *_args, **_kwargs: (model, tokenizer),
    )

    def fake_iter(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(())

    monkeypatch.setattr(streaming_cli, "stream_generate_iter", fake_iter)
    monkeypatch.setattr(sys, "stdout", output)
    _set_cli(monkeypatch, "--tokens", "0", "--incremental-output")

    streaming_cli.main()

    assert output.text == "ab\n"
    assert len(calls) == 1
    assert calls[0][1]["max_new_tokens"] == 0


def test_incremental_output_decodes_bpe_tokens_independently(monkeypatch):
    tokenizer = BPETokenizer(["a", "b", "ab"], [("a", "b")])
    model = _ModelStub(tokenizer.vocab_size)
    output = _RecordingStdout()

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        lambda *_args, **_kwargs: (model, tokenizer),
    )
    monkeypatch.setattr(
        streaming_cli,
        "stream_generate_iter",
        lambda *_args, **_kwargs: iter(
            [np.array([tokenizer.stoi["ab"]], dtype=np.int64)]
        ),
    )
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            "fake.pkl",
            "--prompt",
            "a",
            "--tokens",
            "1",
            "--strategy",
            "greedy",
            "--incremental-output",
        ],
    )

    streaming_cli.main()

    assert output.text == "aab\n"


def test_negative_stop_id_fails_before_checkpoint_io(monkeypatch):
    _set_cli(monkeypatch, "--stop-token-id", "-1")

    def unexpected_checkpoint_read(*_args, **_kwargs):
        raise AssertionError("negative stop ids must fail before checkpoint I/O")

    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        unexpected_checkpoint_read,
    )

    with pytest.raises(ValueError, match="stop-token-id.*non-negative"):
        streaming_cli.main()


def test_out_of_vocab_stop_id_fails_before_prompt_io_or_generation(monkeypatch):
    tokenizer = CharTokenizer.train("abc ")
    model = _ModelStub(tokenizer.vocab_size)
    monkeypatch.setattr(
        streaming_cli,
        "load_streaming_checkpoint",
        lambda *_args, **_kwargs: (model, tokenizer),
    )
    monkeypatch.setattr(
        streaming_cli,
        "stream_generate",
        lambda *_args, **_kwargs: pytest.fail("generation must not start"),
    )
    monkeypatch.setattr(
        streaming_cli,
        "stream_generate_iter",
        lambda *_args, **_kwargs: pytest.fail("generation must not start"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-stream",
            "--checkpoint",
            "fake.pkl",
            "--prompt-file",
            "must-not-be-read.txt",
            "--stop-token-id",
            str(tokenizer.vocab_size),
        ],
    )

    with pytest.raises(ValueError, match=r"stop-token-id.*\[0"):
        streaming_cli.main()


def test_programmatic_stop_id_rejects_boolean_before_io():
    namespace = type(
        "Args",
        (),
        {
            "tokens": 1,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "seed": 7,
            "strategy": "sample",
            "stop_token_id": True,
            "incremental_output": False,
        },
    )()

    with pytest.raises(TypeError, match="stop-token-id"):
        streaming_cli._validate_args(namespace)


def test_programmatic_incremental_flag_requires_boolean():
    namespace = type(
        "Args",
        (),
        {
            "tokens": 1,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "seed": 7,
            "strategy": "sample",
            "stop_token_id": None,
            "incremental_output": "yes",
        },
    )()

    with pytest.raises(TypeError, match="incremental-output"):
        streaming_cli._validate_args(namespace)


def test_numpy_integer_stop_id_is_accepted():
    namespace = type(
        "Args",
        (),
        {
            "tokens": 1,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "seed": 7,
            "strategy": "sample",
            "stop_token_id": np.int64(2),
            "incremental_output": np.bool_(True),
        },
    )()

    streaming_cli._validate_args(namespace)
    assert streaming_cli._validate_stop_token_id(np.int64(2), 4) == 2
