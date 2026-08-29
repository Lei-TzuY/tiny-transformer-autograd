import threading
import time

from engine.metric_moments import WeightedStreamingMoments


def _assert_thread_waits_while_lock_is_held(operation):
    moments = WeightedStreamingMoments()
    moments.update(1.0)
    started = threading.Event()
    finished = threading.Event()

    def worker():
        started.set()
        operation(moments)
        finished.set()

    with moments._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(1.0)
        time.sleep(0.02)
        assert not finished.is_set()

    thread.join(1.0)
    assert not thread.is_alive()
    assert finished.is_set()


def test_update_is_serialized_by_instance_lock():
    _assert_thread_waits_while_lock_is_held(lambda moments: moments.update(2.0))


def test_statistics_is_serialized_by_instance_lock():
    _assert_thread_waits_while_lock_is_held(lambda moments: moments.statistics())


def test_state_dict_is_serialized_by_instance_lock():
    _assert_thread_waits_while_lock_is_held(lambda moments: moments.state_dict())


def test_reset_is_serialized_by_instance_lock():
    _assert_thread_waits_while_lock_is_held(lambda moments: moments.reset())


def test_same_thread_reentrant_public_operations_do_not_deadlock():
    moments = WeightedStreamingMoments()

    with moments._lock:
        moments.update(1.0, weight=2.0)
        moments.update(3.0, weight=1.0)
        stats = moments.statistics()
        state = moments.state_dict()

    assert stats["observation_count"] == 2
    assert state["observation_count"] == 2


def _clone(source):
    clone = WeightedStreamingMoments()
    clone.load_state_dict(source.state_dict())
    return clone


def test_reciprocal_merges_are_deadlock_free_and_linearizable():
    left = WeightedStreamingMoments()
    left.update(-2.0, weight=2.0)
    left.update(3.0, weight=1.0)

    right = WeightedStreamingMoments()
    right.update(7.0, weight=4.0)
    right.update(11.0, weight=1.0)

    original_left = _clone(left)
    original_right = _clone(right)

    expected_ab_left = _clone(original_left)
    expected_ab_right = _clone(original_right)
    expected_ab_left.merge(expected_ab_right)
    expected_ab_right.merge(expected_ab_left)
    expected_ab = (expected_ab_left.state_dict(), expected_ab_right.state_dict())

    expected_ba_left = _clone(original_left)
    expected_ba_right = _clone(original_right)
    expected_ba_right.merge(expected_ba_left)
    expected_ba_left.merge(expected_ba_right)
    expected_ba = (expected_ba_left.state_dict(), expected_ba_right.state_dict())

    barrier = threading.Barrier(3)
    failures = []

    def merge_left():
        try:
            barrier.wait()
            left.merge(right)
        except BaseException as exc:
            failures.append(exc)

    def merge_right():
        try:
            barrier.wait()
            right.merge(left)
        except BaseException as exc:
            failures.append(exc)

    left_thread = threading.Thread(target=merge_left)
    right_thread = threading.Thread(target=merge_right)
    left_thread.start()
    right_thread.start()
    barrier.wait()

    left_thread.join(2.0)
    right_thread.join(2.0)

    assert not left_thread.is_alive()
    assert not right_thread.is_alive()
    assert failures == []
    actual = (left.state_dict(), right.state_dict())
    assert actual in (expected_ab, expected_ba)


def test_merge_with_empty_source_does_not_change_target_under_locking():
    target = WeightedStreamingMoments()
    target.update(5.0, weight=2.0)
    empty = WeightedStreamingMoments()
    before = target.state_dict()

    target.merge(empty)

    assert target.state_dict() == before
