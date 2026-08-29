import threading
import time

from engine.trainability import freeze_, unfreeze_
from engine.tensor import Tensor


def test_complete_trainability_calls_are_serialized_across_threads():
    first = Tensor(1.0, requires_grad=True)
    second = Tensor(2.0, requires_grad=False)
    iterator_entered = threading.Event()
    release_iterator = threading.Event()
    contender_done = threading.Event()
    results = []

    def blocked_parameters():
        iterator_entered.set()
        release_iterator.wait(timeout=5)
        yield first

    def freezer():
        results.append(("freeze", freeze_(blocked_parameters())))

    def contender():
        results.append(("unfreeze", unfreeze_(second)))
        contender_done.set()

    owner = threading.Thread(target=freezer)
    owner.start()
    assert iterator_entered.wait(timeout=5)

    waiting = threading.Thread(target=contender)
    waiting.start()
    time.sleep(0.05)
    assert not contender_done.is_set()
    assert second.requires_grad is False

    release_iterator.set()
    owner.join(timeout=5)
    waiting.join(timeout=5)

    assert not owner.is_alive()
    assert not waiting.is_alive()
    assert first.requires_grad is False
    assert second.requires_grad is True
    assert results == [("freeze", 1), ("unfreeze", 1)]


def test_same_thread_reentrant_helper_call_during_materialization_is_safe():
    outer = Tensor(1.0, requires_grad=True)
    nested = Tensor(2.0, requires_grad=True)
    events = []

    def parameters():
        events.append("before_nested")
        assert freeze_(nested) == 1
        events.append("after_nested")
        yield outer

    assert freeze_(parameters()) == 1

    assert events == ["before_nested", "after_nested"]
    assert outer.requires_grad is False
    assert nested.requires_grad is False


def test_conflicting_calls_on_same_tensor_commit_in_lock_order():
    parameter = Tensor(1.0, requires_grad=True)
    iterator_entered = threading.Event()
    release_iterator = threading.Event()
    second_done = threading.Event()

    def blocked_parameters():
        iterator_entered.set()
        release_iterator.wait(timeout=5)
        yield parameter

    first_thread = threading.Thread(target=lambda: freeze_(blocked_parameters()))

    def second_call():
        unfreeze_(parameter)
        second_done.set()

    first_thread.start()
    assert iterator_entered.wait(timeout=5)
    second_thread = threading.Thread(target=second_call)
    second_thread.start()

    time.sleep(0.05)
    assert not second_done.is_set()
    release_iterator.set()

    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert parameter.requires_grad is True
    assert parameter.grad is None
    assert parameter._version == 2
