"""Integration coverage for GQA/MQA through the packaged training adapter."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train
import training_cli
from engine.checkpoint import read_checkpoint
from engine.safe_checkpoint import read_safe_checkpoint
from nn import GPT


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
        "--seed", "37",
        "--no-sample",
    ]
    if kv_heads is not None:
        argv.extend(["--kv-heads", str(kv_heads)])
    if save is not None:
        argv.extend(["--save", str(save)])
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    return argv


def _assert_state_equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray)
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict)
        assert isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_state_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            _assert_state_equal(l_item, r_item)
        return
    assert left == right


def test_pickle_training_builds_and_persists_gqa(monkeypatch, capsys, tmp_path):
    path = tmp_path / "gqa.pkl"
    monkeypatch.setattr(sys, "argv", _argv("tiny-train", save=path, kv_heads=2))
    training_cli.main()
    capsys.readouterr()

    state = read_checkpoint(path)
    config = state["metadata"]["model_config"]
    assert config["num_heads"] == 4
    assert config["num_kv_heads"] == 2
    assert state["step"] == 1

    model = GPT(**config)
    model.load_state_dict(state["model"], strict=True)
    _, cache = model.infer(np.zeros((1, 2), dtype=np.int64))
    assert cache[0]["k"].shape[1] == 2
    assert cache[0]["v"].shape[1] == 2


def test_safe_training_builds_and_persists_mqa(monkeypatch, capsys, tmp_path):
    path = tmp_path / "mqa.safe.npz"
    monkeypatch.setattr(
        sys, "argv", _argv("tiny-train-safe", save=path, kv_heads=1)
    )
    training_cli.safe_main()
    capsys.readouterr()

    state = read_safe_checkpoint(path)
    config = state["metadata"]["model_config"]
    assert config["num_heads"] == 4
    assert config["num_kv_heads"] == 1
    assert state["step"] == 1


def test_gqa_resume_uses_checkpoint_when_flag_is_omitted(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "resume.pkl"
    monkeypatch.setattr(sys, "argv", _argv("tiny-train", save=path, kv_heads=2))
    training_cli.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        _argv("tiny-train", save=path, resume=path, iters=2, kv_heads=None),
    )
    training_cli.main()
    output = capsys.readouterr().out

    state = read_checkpoint(path)
    assert state["step"] == 2
    assert state["metadata"]["model_config"]["num_kv_heads"] == 2
    assert "resume_step=1" in output


def test_gqa_resume_accepts_explicit_matching_flag(monkeypatch, capsys, tmp_path):
    path = tmp_path / "resume-match.pkl"
    monkeypatch.setattr(sys, "argv", _argv("tiny-train", save=path, kv_heads=2))
    training_cli.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        _argv("tiny-train", save=path, resume=path, iters=2, kv_heads=2),
    )
    training_cli.main()
    capsys.readouterr()

    state = read_checkpoint(path)
    assert state["step"] == 2
    assert state["metadata"]["model_config"]["num_kv_heads"] == 2


def test_legacy_console_path_without_flag_matches_train_main(
    monkeypatch, capsys, tmp_path
):
    direct = tmp_path / "direct.pkl"
    wrapped = tmp_path / "wrapped.pkl"

    monkeypatch.setattr(sys, "argv", _argv("train", save=direct))
    train.main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", _argv("tiny-train", save=wrapped))
    training_cli.main()
    capsys.readouterr()

    direct_state = read_checkpoint(direct)
    wrapped_state = read_checkpoint(wrapped)
    assert "num_kv_heads" not in direct_state["metadata"]["model_config"]
    assert "num_kv_heads" not in wrapped_state["metadata"]["model_config"]
    _assert_state_equal(direct_state, wrapped_state)
