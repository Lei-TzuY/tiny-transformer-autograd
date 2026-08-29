import threading
import time

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


def test_merge_does_not_hold_source_lock_while_waiting_for_target_lock():
    source = WeightedMetricAccumulator()
    target = WeightedMetricAccumulator()
    source.update(7.0)
    target.update(1.0)

    started = threading.Event()
    finished = threading.Event()

    def merger():
        started.set()
        target.merge(source)
        finished.set()

    with target._lock:
        thread = threading.Thread(target=merger)
        thread.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.05)
        assert not finished.is_set()

        # The merge has already snapshotted and released the source lock before
        # waiting for the target, so unrelated source reads remain available.
        assert source.state_dict()["mean"] == 7.0

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert finished.is_set()
    assert target.mean == 4.0
