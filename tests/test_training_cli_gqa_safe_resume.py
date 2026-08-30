"""Safe-checkpoint resume coverage for the GQA-aware training entry point."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import training_cli
from engine.safe_checkpoint import read_safe_checkpoint


def _argv(*, path, iters=1, resume=False, kv_heads=None):
    argv = [
        "tiny-train-safe",
        "--iters", str(iters),
        "--eval-interval", "1",
        "--eval-iters", "1",
        "--ctx", "4",
        "--d", "8",
        "--heads", "4",
        "--layers", "1",
        "--batch", "2",
        "--min-lr", "0.0001",
        "--seed", "43",
        "--no-sample",
        "--save", str(path),
    ]
    if resume:
        argv.extend(["--resume", str(path)])
    if kv_heads is not None:
        argv.extend(["--kv-heads", str(kv_heads)])
    return argv


def test_safe_gqa_resume_without_flag_preserves_architecture(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "resume.safe.npz"
    monkeypatch.setattr(sys, "argv", _argv(path=path, kv_heads=2))
    training_cli.safe_main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", _argv(path=path, iters=2, resume=True))
    training_cli.safe_main()
    capsys.readouterr()

    state = read_safe_checkpoint(path)
    assert state["step"] == 2
    assert state["metadata"]["model_config"]["num_kv_heads"] == 2


def test_safe_gqa_resume_conflict_is_state_neutral(monkeypatch, capsys, tmp_path):
    path = tmp_path / "conflict.safe.npz"
    monkeypatch.setattr(sys, "argv", _argv(path=path, kv_heads=2))
    training_cli.safe_main()
    capsys.readouterr()
    before = path.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(path=path, iters=2, resume=True, kv_heads=1),
    )
    with pytest.raises(ValueError, match="conflicts with checkpoint num_kv_heads 2"):
        training_cli.safe_main()
    assert path.read_bytes() == before
