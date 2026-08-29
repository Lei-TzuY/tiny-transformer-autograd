import threading
import time

import pytest

import engine.early_stopping as early_stopping_module
from engine.early_stopping import EarlyStopping


def test_state_reads_wait_for_in_progress_step(monkeypatch):
    stopper = EarlyStopping(mode="min", patience=2)
    stopper.step(5.0)

    entered = threading.Event()
    release = threading.Event()
    reader_done = threading.Event()
    original = early_stopping_module._is_improvement

    def blocking_is_improvement(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(early_stopping_module, "_is_improvement", blocking_is_improvement)

    step_result = []
    reader_result = []
    step_thread = threading.Thread(target=lambda: step_result.append(stopper.step(6.0)))

    def read_state():
        reader_result.append(stopper.state_dict())
        reader_done.set()

    reader_thread = threading.Thread(target=read_state)
    step_thread.start()
    assert entered.wait(timeout=5)
    reader_thread.start()

    time.sleep(0.05)
    assert not reader_done.is_set()

    release.set()
    step_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert not step_thread.is_alive()
    assert not reader_thread.is_alive()

    assert step_result == [False]
    assert reader_result[0]["step_count"] == 2
    assert reader_result[0]["num_bad_epochs"] == 1


def test_same_thread_reentrant_state_read_during_step(monkeypatch):
    stopper = EarlyStopping(mode="max", patience=1)
    stopper.step(1.0)
    original = early_stopping_module._is_improvement
    seen = []

    def reentrant_is_improvement(*args, **kwargs):
        seen.append(stopper.state_dict()["step_count"])
        return original(*args, **kwargs)

    monkeypatch.setattr(early_stopping_module, "_is_improvement", reentrant_is_improvement)
    assert stopper.step(2.0) is False
    assert seen == [1]
    assert stopper.best == 2.0


def test_load_state_waits_for_in_progress_step(monkeypatch):
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    replacement = EarlyStopping(mode="max", patience=4, min_delta=0.5).state_dict()

    entered = threading.Event()
    release = threading.Event()
    loaded = threading.Event()
    original = early_stopping_module._is_improvement

    def blocking_is_improvement(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(early_stopping_module, "_is_improvement", blocking_is_improvement)
    worker = threading.Thread(target=lambda: stopper.step(2.0))

    def load():
        stopper.load_state_dict(replacement)
        loaded.set()

    loader = threading.Thread(target=load)
    worker.start()
    assert entered.wait(timeout=5)
    loader.start()
    time.sleep(0.05)
    assert not loaded.is_set()

    release.set()
    worker.join(timeout=5)
    loader.join(timeout=5)
    assert loaded.is_set()
    assert stopper.mode == "max"
    assert stopper.patience == 4
    assert stopper.step_count == 0


def test_reset_is_reentrant_under_explicit_lock():
    stopper = EarlyStopping(patience=1)
    stopper.step(1.0)
    with stopper._lock:
        assert stopper.reset() is stopper
        assert stopper.state_dict()["step_count"] == 0
