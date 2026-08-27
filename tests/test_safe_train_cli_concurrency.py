"""Isolation regressions for overlapping in-process safe training calls."""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import safe_train_cli
import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


def test_overlapping_safe_calls_serialize_checkpoint_io_swap(monkeypatch):
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()
    results = []
    errors = []

    def fake_main():
        name = threading.current_thread().name
        assert train.read_checkpoint is read_safe_checkpoint
        assert train.save_checkpoint is save_safe_checkpoint
        if name == "safe-first":
            first_inside.set()
            if not release_first.wait(timeout=2.0):
                raise AssertionError("first safe call was not released")
        else:
            second_inside.set()
        return name

    def run():
        try:
            results.append(safe_train_cli.main())
        except BaseException as exc:  # propagate worker failures in the main test thread
            errors.append(exc)

    monkeypatch.setattr(train, "main", fake_main)

    first = threading.Thread(target=run, name="safe-first")
    second = threading.Thread(target=run, name="safe-second")
    first.start()
    assert first_inside.wait(timeout=2.0)

    second.start()
    # The first invocation deliberately remains inside train.main. The second
    # must not reach it while the temporary module-global I/O swap is owned by
    # the first invocation.
    assert not second_inside.wait(timeout=0.15)

    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sorted(results) == ["safe-first", "safe-second"]
    assert second_inside.is_set()
    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer


def test_same_thread_nested_safe_call_remains_reentrant(monkeypatch):
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint
    calls = []

    def fake_main():
        calls.append((train.read_checkpoint, train.save_checkpoint))
        if len(calls) == 1:
            return safe_train_cli.main()
        return "nested"

    monkeypatch.setattr(train, "main", fake_main)

    assert safe_train_cli.main() == "nested"
    assert calls == [
        (read_safe_checkpoint, save_safe_checkpoint),
        (read_safe_checkpoint, save_safe_checkpoint),
    ]
    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer
