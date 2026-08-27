"""Integration tests for the non-executable safe training entry point."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import safe_train_cli
import train
from engine.checkpoint import read_checkpoint
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


def _tiny_train_argv(program, *, iters=1, save=None, resume=None):
    argv = [
        program,
        "--iters", str(iters),
        "--eval-interval", "1",
        "--eval-iters", "1",
        "--ctx", "4",
        "--d", "4",
        "--heads", "1",
        "--layers", "1",
        "--batch", "2",
        "--seed", "23",
        "--no-sample",
    ]
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
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_state_equal(left_item, right_item)
        return
    assert left == right


def _first_model_array(state):
    for value in state["model"].values():
        if isinstance(value, np.ndarray) and value.size:
            return value
    raise AssertionError("model checkpoint contained no parameter arrays")


def test_adapter_swaps_checkpoint_io_only_during_successful_call(monkeypatch):
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint
    observed = []

    def fake_main():
        observed.append((train.read_checkpoint, train.save_checkpoint))
        return "done"

    monkeypatch.setattr(train, "main", fake_main)

    assert safe_train_cli.main() == "done"
    assert observed == [(read_safe_checkpoint, save_safe_checkpoint)]
    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer


def test_adapter_restores_checkpoint_io_after_failure(monkeypatch):
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint

    def fail():
        assert train.read_checkpoint is read_safe_checkpoint
        assert train.save_checkpoint is save_safe_checkpoint
        raise RuntimeError("training failed")

    monkeypatch.setattr(train, "main", fail)

    with pytest.raises(RuntimeError, match="training failed"):
        safe_train_cli.main()

    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer


def test_pickle_and_safe_cli_save_identical_training_state(
    monkeypatch, capsys, tmp_path
):
    pickle_path = tmp_path / "run.pkl"
    safe_path = tmp_path / "run.safe.npz"

    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv("tiny-train", save=pickle_path),
    )
    train.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv("tiny-train-safe", save=safe_path),
    )
    safe_train_cli.main()
    capsys.readouterr()

    pickle_state = read_checkpoint(pickle_path)
    safe_state = read_safe_checkpoint(safe_path)
    _assert_state_equal(pickle_state, safe_state)


def test_safe_cli_resumes_and_rewrites_safe_checkpoint(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "resume.safe.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv("tiny-train-safe", iters=1, save=path),
    )
    safe_train_cli.main()
    capsys.readouterr()

    first_state = read_safe_checkpoint(path)
    first_parameter = _first_model_array(first_state).copy()
    assert first_state["step"] == 1
    assert first_state["optimizer_type"] == "Adam"

    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv(
            "tiny-train-safe",
            iters=2,
            save=path,
            resume=path,
        ),
    )
    safe_train_cli.main()
    output = capsys.readouterr().out

    resumed = read_safe_checkpoint(path)
    assert resumed["step"] == 2
    assert resumed["optimizer_type"] == "Adam"
    assert "resume_step=1" in output
    assert not np.array_equal(_first_model_array(resumed), first_parameter)


def test_safe_cli_generate_only_reads_safe_checkpoint(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "generate.safe.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv("tiny-train-safe", save=path),
    )
    safe_train_cli.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-train-safe",
            "--resume", str(path),
            "--generate-only",
            "--sample", "0",
            "--prompt", "To",
            "--seed", "23",
        ],
    )
    safe_train_cli.main()
    output = capsys.readouterr().out

    assert "Sample generation:" in output
    assert "To" in output


def test_safe_cli_never_falls_back_to_pickle_resume(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "trusted.pkl"
    monkeypatch.setattr(
        sys,
        "argv",
        _tiny_train_argv("tiny-train", save=path),
    )
    train.main()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny-train-safe",
            "--resume", str(path),
            "--generate-only",
            "--sample", "0",
        ],
    )
    with pytest.raises(ValueError, match="invalid safe checkpoint container"):
        safe_train_cli.main()
