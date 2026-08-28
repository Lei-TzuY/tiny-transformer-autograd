"""Concurrency regression tests for the safe training adapter."""

import threading

import safe_train_cli
import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


def test_concurrent_safe_calls_cannot_restore_pickle_io_early(monkeypatch):
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint
    first_inside = threading.Event()
    release_first = threading.Event()
    first_returned = threading.Event()
    second_inside = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    observations = []
    results = []
    errors = []

    def fake_main():
        nonlocal call_count
        with call_lock:
            call_count += 1
            invocation = call_count

        assert train.read_checkpoint is read_safe_checkpoint
        assert train.save_checkpoint is save_safe_checkpoint
        if invocation == 1:
            first_inside.set()
            assert release_first.wait(timeout=2.0)
        else:
            second_inside.set()
            assert first_returned.wait(timeout=2.0)
            observations.append((train.read_checkpoint, train.save_checkpoint))
            assert train.read_checkpoint is read_safe_checkpoint
            assert train.save_checkpoint is save_safe_checkpoint
        return invocation

    monkeypatch.setattr(train, "main", fake_main)

    def run(target, returned=None):
        try:
            results.append(target())
        except BaseException as exc:  # surface worker failures in the main test
            errors.append(exc)
        finally:
            if returned is not None:
                returned.set()

    first = threading.Thread(
        target=run,
        args=(safe_train_cli.main, first_returned),
    )
    second = threading.Thread(target=run, args=(safe_train_cli.main,))

    first.start()
    assert first_inside.wait(timeout=2.0)
    second.start()

    # Give the second worker a real opportunity to enter. It must remain
    # outside train.main() until the first process-global patch is restored.
    overlapped = second_inside.wait(timeout=0.25)
    release_first.set()

    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not overlapped
    assert errors == []
    assert results == [1, 2]
    assert observations == [(read_safe_checkpoint, save_safe_checkpoint)]
    assert train.read_checkpoint is original_reader
    assert train.save_checkpoint is original_writer
