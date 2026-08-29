import threading

import numpy as np

from engine.gradient_value_clip import clip_grad_value_
from engine.tensor import Tensor


class _MutateThenBlockAndFail(np.ndarray):
    def __new__(cls, values, entered, release):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.entered = entered
        obj.release = release
        obj.failed_once = False
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self.entered = getattr(source, "entered", None)
        self.release = getattr(source, "release", None)
        self.failed_once = getattr(source, "failed_once", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not self.failed_once:
            self.failed_once = True
            self.entered.set()
            if not self.release.wait(timeout=5.0):
                raise AssertionError("timed out waiting to release injected gradient write")
            raise RuntimeError("injected gradient write failure")


class _ReentrantGradient(np.ndarray):
    def __new__(cls, values, nested_parameter):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.nested_parameter = nested_parameter
        obj.called = False
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self.nested_parameter = getattr(source, "nested_parameter", None)
        self.called = getattr(source, "called", False)

    def __setitem__(self, key, value):
        if not self.called:
            self.called = True
            assert clip_grad_value_(self.nested_parameter, 1.0) == 1
        super().__setitem__(key, value)


def test_failed_transaction_cannot_rollback_over_concurrent_success():
    first = Tensor(np.array([0.0]), requires_grad=True)
    failing = Tensor(np.array([0.0]), requires_grad=True)
    first.grad = np.array([5.0])

    entered = threading.Event()
    release = threading.Event()
    failing.grad = _MutateThenBlockAndFail([5.0], entered, release)

    first_error = []
    second_result = []
    second_done = threading.Event()

    def failing_worker():
        try:
            clip_grad_value_([first, failing], 2.0)
        except BaseException as exc:
            first_error.append(exc)

    def succeeding_worker():
        second_result.append(clip_grad_value_(first, 1.0))
        second_done.set()

    worker = threading.Thread(target=failing_worker)
    worker.start()
    assert entered.wait(timeout=5.0)

    # The first transaction has already written first.grad=2. A concurrent helper call
    # must not observe that dirty value or commit work that the first rollback can erase.
    contender = threading.Thread(target=succeeding_worker)
    contender.start()
    assert not second_done.wait(timeout=0.1)

    release.set()
    worker.join(timeout=5.0)
    contender.join(timeout=5.0)

    assert not worker.is_alive()
    assert not contender.is_alive()
    assert len(first_error) == 1
    assert isinstance(first_error[0], RuntimeError)
    assert str(first_error[0]) == "injected gradient write failure"
    assert second_result == [1]
    np.testing.assert_array_equal(first.grad, np.array([1.0]))
    np.testing.assert_array_equal(failing.grad, np.array([5.0]))


def test_transaction_lock_is_same_thread_reentrant():
    nested = Tensor(np.array([0.0]), requires_grad=True)
    nested.grad = np.array([3.0])

    outer = Tensor(np.array([0.0]), requires_grad=True)
    outer.grad = _ReentrantGradient([4.0], nested)

    assert clip_grad_value_(outer, 2.0) == 1
    np.testing.assert_array_equal(outer.grad, np.array([2.0]))
    np.testing.assert_array_equal(nested.grad, np.array([1.0]))
