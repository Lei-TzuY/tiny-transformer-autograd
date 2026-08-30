"""Validation, conflict, help, and cleanup tests for GQA training CLIs."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train
import training_cli
from engine.checkpoint import read_checkpoint


def _argv(program, *, save=None, resume=None, iters=1, kv_heads=None):
    argv = [
        program,
        "--iters", str(iters),
        "--eval-interval", "1",
        "--eval-iters", "1",
        "--ctx", "4",
        "--d", "8",
        "--heads", "4",
        "--layers", "1",
        "--batch", "2",
        "--min-lr", "0.0001",
        "--seed", "41",
        "--no-sample",
    ]
    if kv_heads is not None:
        argv.extend(["--kv-heads", str(kv_heads)])
    if save is not None:
        argv.extend(["--save", str(save)])
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    return argv


def test_extract_kv_heads_supports_split_and_equals_forms():
    stripped, requested = training_cli._extract_kv_heads(
        ["tiny-train", "--kv-heads", "2", "--iters", "3"]
    )
    assert stripped == ["tiny-train", "--iters", "3"]
    assert requested == 2

    stripped, requested = training_cli._extract_kv_heads(
        ["tiny-train", "--iters", "3", "--kv-heads=1"]
    )
    assert stripped == ["tiny-train", "--iters", "3"]
    assert requested == 1


@pytest.mark.parametrize("argv", [
    ["tiny-train", "--kv-heads", "0"],
    ["tiny-train", "--kv-heads", "-1"],
    ["tiny-train", "--kv-heads", "abc"],
    ["tiny-train", "--kv-heads"],
])
def test_extract_kv_heads_rejects_invalid_values(argv):
    with pytest.raises(SystemExit) as exc_info:
        training_cli._extract_kv_heads(argv)
    assert exc_info.value.code == 2


def test_new_run_rejects_nondividing_kv_heads(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("tiny-train", save=tmp_path / "bad.pkl", kv_heads=3),
    )
    with pytest.raises(
        ValueError, match=r"--kv-heads 3 must divide model num_heads 4"
    ):
        training_cli.main()


def test_resume_conflict_fails_before_checkpoint_rewrite(monkeypatch, capsys, tmp_path):
    path = tmp_path / "conflict.pkl"
    monkeypatch.setattr(sys, "argv", _argv("tiny-train", save=path, kv_heads=2))
    training_cli.main()
    capsys.readouterr()
    before = path.read_bytes()
    assert read_checkpoint(path)["metadata"]["model_config"]["num_kv_heads"] == 2

    monkeypatch.setattr(
        sys,
        "argv",
        _argv("tiny-train", save=path, resume=path, iters=2, kv_heads=1),
    )
    with pytest.raises(
        ValueError,
        match=r"--kv-heads 1 conflicts with checkpoint num_kv_heads 2",
    ):
        training_cli.main()
    assert path.read_bytes() == before


def test_checkpoint_kv_head_validation_rejects_bool_and_bad_partition():
    with pytest.raises(TypeError, match="num_heads must be an integer"):
        training_cli._checkpoint_kv_heads(
            {"metadata": {"model_config": {"num_heads": True}}}
        )
    with pytest.raises(TypeError, match="num_kv_heads must be an integer"):
        training_cli._checkpoint_kv_heads(
            {"metadata": {"model_config": {"num_heads": 4, "num_kv_heads": False}}}
        )
    with pytest.raises(ValueError, match="divisible by num_kv_heads"):
        training_cli._checkpoint_kv_heads(
            {"metadata": {"model_config": {"num_heads": 4, "num_kv_heads": 3}}}
        )


def test_adapter_restores_train_globals_and_argv_after_failure(monkeypatch):
    original_argv = ["tiny-train", "--kv-heads", "2"]
    monkeypatch.setattr(sys, "argv", original_argv)
    original_gpt = train.GPT
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint

    def fail():
        assert train.GPT is not original_gpt
        assert train.read_checkpoint is not original_reader
        assert train.save_checkpoint is original_writer
        assert sys.argv == ["tiny-train"]
        raise RuntimeError("injected failure")

    monkeypatch.setattr(train, "main", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        training_cli.main()

    assert sys.argv is original_argv
    assert train.GPT is original_gpt
    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer


def test_installed_training_console_help_documents_kv_heads():
    for command in ("tiny-train", "tiny-train-safe"):
        executable = shutil.which(command)
        assert executable is not None, f"installed console script {command!r} was not found"
        completed = subprocess.run(
            [executable, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout.lower()
        assert "--kv-heads" in completed.stdout
        assert completed.stderr == ""
