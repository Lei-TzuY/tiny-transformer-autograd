import threading
import time

import pytest

from engine.metric_accumulator import WeightedMetricAccumulator


def test_update_waits_for_in_progress_operation_on_same_accumulator():
    meter = WeightedMetricAccumulator()
    started = threading.Event()
    finished = threading.Event()

    def worker():
        started.set()
        meter.update(3.0, weight=2.0)
        finished.set()

    with meter._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not finished.is_set()

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert finished.is_set()
    assert meter.mean == 3.0


def test_state_read_waits_for_same_instance_lock():
    meter = WeightedMetricAccumulator()
    meter.update(5.0)
    started = threading.Event()
    finished = threading.Event()
    observed = []

    def reader():
        started.set()
        observed.append(meter.state_dict())
        finished.set()

    with meter._lock:
        thread = threading.Thread(target=reader)
        thread.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not finished.is_set()

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert observed[0]["mean"] == 5.0


def test_reset_waits_for_same_instance_lock():
    meter = WeightedMetricAccumulator()
    meter.update(5.0)
    started = threading.Event()
    finished = threading.Event()

    def resetter():
        started.set()
        meter.reset()
        finished.set()

    with meter._lock:
        thread = threading.Thread(target=resetter)
        thread.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not finished.is_set()

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert meter.mean is None
    assert meter.observation_count == 0


def test_same_thread_operations_are_reentrant():
    meter = WeightedMetricAccumulator()

    with meter._lock:
        meter.update(2.0, weight=3.0)
        state = meter.state_dict()
        assert meter.mean == 2.0
        assert meter.total_weight == 3.0
        assert meter.observation_count == 1
        meter.load_state_dict(state)

    assert meter.state_dict() == state


def test_merge_holds_source_while_waiting_for_target_when_source_orders_first():
    first = WeightedMetricAccumulator()
    second = WeightedMetricAccumulator()
    source, target = sorted((first, second), key=id)
    source.update(7.0)
    target.update(1.0)

    merge_started = threading.Event()
    merge_finished = threading.Event()
    update_started = threading.Event()
    update_finished = threading.Event()

    def merger():
        merge_started.set()
        target.merge(source)
        merge_finished.set()

    def updater():
        update_started.set()
        source.update(9.0)
        update_finished.set()

    with target._lock:
        merge_thread = threading.Thread(target=merger)
        merge_thread.start()
        assert merge_started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not merge_finished.is_set()

        update_thread = threading.Thread(target=updater)
        update_thread.start()
        assert update_started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not update_finished.is_set()

    merge_thread.join(timeout=1.0)
    update_thread.join(timeout=1.0)
    assert not merge_thread.is_alive()
    assert not update_thread.is_alive()
    assert target.mean == 4.0
    assert source.mean == 8.0


def test_reciprocal_merges_are_deadlock_free_and_linearizable():
    barrier = threading.Barrier(2)

    class SnapshotBarrierAccumulator(WeightedMetricAccumulator):
        def state_dict(self):
            state = super().state_dict()
            barrier.wait(timeout=1.0)
            return state

    left = SnapshotBarrierAccumulator()
    right = SnapshotBarrierAccumulator()
    left.update(1.0)
    right.update(3.0)
    failures = []

    def merge_left():
        try:
            left.merge(right)
        except BaseException as exc:
            failures.append(exc)

    def merge_right():
        try:
            right.merge(left)
        except BaseException as exc:
            failures.append(exc)

    left_thread = threading.Thread(target=merge_left)
    right_thread = threading.Thread(target=merge_right)
    left_thread.start()
    right_thread.start()
    left_thread.join(timeout=1.0)
    right_thread.join(timeout=1.0)

    assert not left_thread.is_alive()
    assert not right_thread.is_alive()
    assert failures == []

    outcome = (
        left.observation_count,
        left.total_weight,
        left.mean,
        right.observation_count,
        right.total_weight,
        right.mean,
    )
    serial_left_first = (2, 2.0, 2.0, 3, 3.0, pytest.approx(7.0 / 3.0))
    serial_right_first = (3, 3.0, pytest.approx(5.0 / 3.0), 2, 2.0, 2.0)
    assert outcome == serial_left_first or outcome == serial_right_first
