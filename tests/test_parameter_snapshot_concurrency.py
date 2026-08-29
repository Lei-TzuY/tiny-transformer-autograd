import threading

import numpy as np

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_cross_instance_capture_waits_for_temporary_scope_and_sees_restored_values():
    p = Tensor([3.0], requires_grad=True)
    temporary = ParameterSnapshot(p, values=np.array([1.0]))
    observer = ParameterSnapshot(p)
    entered = threading.Event()
    release = threading.Event()
    capture_started = threading.Event()
    capture_finished = threading.Event()

    def hold_temporary():
        with temporary.installed():
            np.testing.assert_array_equal(p.data, [1.0])
            entered.set()
            assert release.wait(timeout=5)

    def capture_after_enter():
        assert entered.wait(timeout=5)
        capture_started.set()
        observer.capture()
        capture_finished.set()

    holder = threading.Thread(target=hold_temporary)
    reader = threading.Thread(target=capture_after_enter)
    holder.start()
    reader.start()

    assert capture_started.wait(timeout=5)
    assert not capture_finished.wait(timeout=0.1)
    release.set()
    holder.join(timeout=5)
    reader.join(timeout=5)

    assert not holder.is_alive()
    assert not reader.is_alive()
    np.testing.assert_array_equal(p.data, [3.0])
    np.testing.assert_array_equal(observer.values()[0], [3.0])


def test_cross_instance_restore_waits_for_temporary_scope_then_commits_after_restore():
    p = Tensor([3.0], requires_grad=True)
    temporary = ParameterSnapshot(p, values=np.array([1.0]))
    contender = ParameterSnapshot(p, values=np.array([2.0]))
    entered = threading.Event()
    release = threading.Event()
    restore_started = threading.Event()
    restore_finished = threading.Event()

    def hold_temporary():
        with temporary.installed():
            np.testing.assert_array_equal(p.data, [1.0])
            entered.set()
            assert release.wait(timeout=5)

    def restore_after_enter():
        assert entered.wait(timeout=5)
        restore_started.set()
        contender.restore()
        restore_finished.set()

    holder = threading.Thread(target=hold_temporary)
    writer = threading.Thread(target=restore_after_enter)
    holder.start()
    writer.start()

    assert restore_started.wait(timeout=5)
    assert not restore_finished.wait(timeout=0.1)
    release.set()
    holder.join(timeout=5)
    writer.join(timeout=5)

    assert not holder.is_alive()
    assert not writer.is_alive()
    np.testing.assert_array_equal(p.data, [2.0])


def test_same_thread_snapshot_operations_are_reentrant_inside_installed_scope():
    p = Tensor([3.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([1.0]))

    with snapshot.installed():
        assert snapshot.parameter_count == 1
        np.testing.assert_array_equal(snapshot.values()[0], [1.0])
        state = snapshot.state_dict()
        assert state["type"] == "ParameterSnapshot"

    np.testing.assert_array_equal(p.data, [3.0])


def test_nested_different_snapshot_instances_do_not_deadlock_same_thread():
    p = Tensor([3.0], requires_grad=True)
    first = ParameterSnapshot(p, values=np.array([1.0]))
    second = ParameterSnapshot(p, values=np.array([2.0]))

    with first.installed():
        np.testing.assert_array_equal(p.data, [1.0])
        with second.installed():
            np.testing.assert_array_equal(p.data, [2.0])
        np.testing.assert_array_equal(p.data, [1.0])

    np.testing.assert_array_equal(p.data, [3.0])


def test_two_independent_restore_calls_are_linearized():
    p = Tensor([0.0], requires_grad=True)
    first = ParameterSnapshot(p, values=np.array([1.0]))
    second = ParameterSnapshot(p, values=np.array([2.0]))
    barrier = threading.Barrier(3)
    completed = []

    def restore(snapshot, label):
        barrier.wait(timeout=5)
        snapshot.restore()
        completed.append(label)

    one = threading.Thread(target=restore, args=(first, "first"))
    two = threading.Thread(target=restore, args=(second, "second"))
    one.start()
    two.start()
    barrier.wait(timeout=5)
    one.join(timeout=5)
    two.join(timeout=5)

    assert not one.is_alive()
    assert not two.is_alive()
    assert sorted(completed) == ["first", "second"]
    assert p.data.item() in (1.0, 2.0)
