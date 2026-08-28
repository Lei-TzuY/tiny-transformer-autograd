import threading

import numpy as np
import pytest

from engine.plateau import ReduceLROnPlateau


class GuardedOptimizer:
    def __init__(self, lr):
        self._lr = float(lr)
        self.fail_target = None
        self.mutate_before_failure = False
        self.block_target = None
        self.setter_entered = threading.Event()
        self.setter_release = threading.Event()

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        value = float(value)
        if self.block_target is not None and value == self.block_target:
            self.setter_entered.set()
            if not self.setter_release.wait(timeout=5):
                raise RuntimeError("timed out waiting to release LR setter")
        if self.fail_target is not None and value == self.fail_target:
            if self.mutate_before_failure:
                self._lr = value
            raise RuntimeError("injected LR setter failure")
        self._lr = value


def _ready_for_reduction(optimizer):
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        eps=0.0,
    )
    scheduler.step(1.0)
    return scheduler


def test_step_setter_failure_after_partial_external_mutation_rolls_lr_back():
    optimizer = GuardedOptimizer(1.0)
    scheduler = _ready_for_reduction(optimizer)
    baseline = scheduler.state_dict()
    optimizer.fail_target = 0.5
    optimizer.mutate_before_failure = True

    with pytest.raises(RuntimeError, match="injected LR setter failure"):
        scheduler.step(2.0)

    assert optimizer.lr == 1.0
    assert scheduler.state_dict() == baseline


def test_load_setter_failure_after_partial_external_mutation_is_fully_neutral():
    source_optimizer = GuardedOptimizer(1.0)
    source = _ready_for_reduction(source_optimizer)
    source.step(2.0)
    state = source.state_dict()

    target_optimizer = GuardedOptimizer(0.8)
    target = ReduceLROnPlateau(target_optimizer)
    baseline = target.state_dict()
    target_optimizer.fail_target = state["current_lr"]
    target_optimizer.mutate_before_failure = True

    with pytest.raises(RuntimeError, match="injected LR setter failure"):
        target.load_state_dict(state)

    assert target_optimizer.lr == baseline["current_lr"]
    assert target.state_dict() == baseline


def test_failed_lr_rollback_is_reported_explicitly():
    class BrokenRollbackOptimizer:
        def __init__(self):
            self._lr = 1.0
            self.writes = 0

        @property
        def lr(self):
            return self._lr

        @lr.setter
        def lr(self, value):
            self.writes += 1
            self._lr = float(value)
            raise RuntimeError(f"write {self.writes} failed")

    optimizer = BrokenRollbackOptimizer()
    scheduler = _ready_for_reduction(optimizer)

    with pytest.raises(RuntimeError, match="optimizer lr rollback failed"):
        scheduler.step(2.0)


def test_subnormal_reduction_never_writes_zero_lr():
    tiny = np.nextafter(0.0, 1.0)
    optimizer = GuardedOptimizer(tiny * 4.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.0,
        eps=0.0,
    )
    scheduler.step(1.0)

    with np.errstate(all="raise"):
        lr = scheduler.step(2.0)

    assert lr == tiny
    assert optimizer.lr == tiny
    assert scheduler.get_lr() == tiny


def test_scheduler_lock_serializes_state_reads_behind_lr_commit():
    optimizer = GuardedOptimizer(1.0)
    scheduler = _ready_for_reduction(optimizer)
    optimizer.block_target = 0.5

    failures = []
    reduction_done = threading.Event()
    reader_entered = threading.Event()
    reader_done = threading.Event()
    observed = []

    def reduce_lr():
        try:
            scheduler.step(2.0)
        except BaseException as exc:  # surfaced in the main test thread
            failures.append(exc)
        finally:
            reduction_done.set()

    def read_state():
        reader_entered.set()
        try:
            observed.append(scheduler.state_dict())
        except BaseException as exc:
            failures.append(exc)
        finally:
            reader_done.set()

    first = threading.Thread(target=reduce_lr)
    first.start()
    assert optimizer.setter_entered.wait(timeout=5)

    second = threading.Thread(target=read_state)
    second.start()
    assert reader_entered.wait(timeout=5)
    assert not reader_done.wait(timeout=0.05)

    optimizer.setter_release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert reduction_done.is_set()
    assert reader_done.is_set()
    assert observed[0]["current_lr"] == 0.5
    assert observed[0]["reductions"] == 1


def test_same_thread_lock_is_reentrant_during_custom_lr_setter():
    class ReentrantOptimizer:
        def __init__(self):
            self._lr = 1.0
            self.scheduler = None
            self.observed_step_count = None

        @property
        def lr(self):
            return self._lr

        @lr.setter
        def lr(self, value):
            # The setter runs while scheduler.step() holds its RLock. Reading a
            # scheduler property from this same thread must not deadlock.
            if self.scheduler is not None:
                self.observed_step_count = self.scheduler.step_count
            self._lr = float(value)

    optimizer = ReentrantOptimizer()
    scheduler = _ready_for_reduction(optimizer)
    optimizer.scheduler = scheduler

    scheduler.step(2.0)

    assert optimizer.observed_step_count == 1
    assert optimizer.lr == 0.5
