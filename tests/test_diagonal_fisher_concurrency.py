import threading

import numpy as np

from engine.diagonal_fisher import DiagonalFisherEstimator
from engine.tensor import Tensor


class BlockingFiniteArray(np.ndarray):
    def __new__(cls, values, entered, release):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.entered = entered
        obj.release = release
        obj.blocked = False
        return obj

    def __array_finalize__(self, source):
        self.entered = getattr(source, "entered", None)
        self.release = getattr(source, "release", None)
        self.blocked = getattr(source, "blocked", False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.isfinite and not self.blocked:
            self.blocked = True
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting to release capture")
        raw = tuple(np.asarray(value) if isinstance(value, BlockingFiniteArray) else value for value in inputs)
        return getattr(ufunc, method)(*raw, **kwargs)


class ReentrantFiniteArray(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.estimator = None
        obj.reentered = False
        return obj

    def __array_finalize__(self, source):
        self.estimator = getattr(source, "estimator", None)
        self.reentered = getattr(source, "reentered", False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.isfinite and self.estimator is not None and not self.reentered:
            self.reentered = True
            assert self.estimator.observation_count == 0
        raw = tuple(np.asarray(value) if isinstance(value, ReentrantFiniteArray) else value for value in inputs)
        return getattr(ufunc, method)(*raw, **kwargs)


def test_state_reader_waits_until_capture_commits_complete_observation():
    entered = threading.Event()
    release = threading.Event()
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = BlockingFiniteArray([2.0], entered, release)
    estimator = DiagonalFisherEstimator(parameter)
    capture_error = []
    reader_result = []

    def capture():
        try:
            estimator.capture()
        except BaseException as exc:  # pragma: no cover - failure diagnostic
            capture_error.append(exc)

    def read_state():
        reader_result.append(estimator.state_dict())

    capture_thread = threading.Thread(target=capture)
    capture_thread.start()
    assert entered.wait(timeout=5)

    reader_thread = threading.Thread(target=read_state)
    reader_thread.start()
    reader_thread.join(timeout=0.1)
    assert reader_thread.is_alive()

    release.set()
    capture_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert not capture_thread.is_alive()
    assert not reader_thread.is_alive()
    assert capture_error == []
    assert reader_result[0]["observation_count"] == 1
    assert reader_result[0]["total_weight"] == 1.0


def test_report_and_reset_wait_behind_in_progress_capture():
    entered = threading.Event()
    release = threading.Event()
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = BlockingFiniteArray([3.0], entered, release)
    estimator = DiagonalFisherEstimator(parameter)
    reports = []

    capture_thread = threading.Thread(target=estimator.capture)
    capture_thread.start()
    assert entered.wait(timeout=5)

    report_thread = threading.Thread(target=lambda: reports.append(estimator.trace_report()))
    reset_thread = threading.Thread(target=estimator.reset)
    report_thread.start()
    reset_thread.start()
    report_thread.join(timeout=0.1)
    reset_thread.join(timeout=0.1)
    assert report_thread.is_alive() or reset_thread.is_alive()

    release.set()
    capture_thread.join(timeout=5)
    report_thread.join(timeout=5)
    reset_thread.join(timeout=5)
    assert not capture_thread.is_alive()
    assert not report_thread.is_alive()
    assert not reset_thread.is_alive()
    # The two queued operations may run in either order after capture, but each
    # must observe a complete state rather than a half-committed sample.
    assert reports[0]["observation_count"] in (0, 1)
    assert estimator.observation_count == 0


def test_same_thread_gradient_validation_can_reenter_estimator_lock():
    parameter = Tensor([0.0], requires_grad=True)
    gradient = ReentrantFiniteArray([2.0])
    parameter.grad = gradient
    estimator = DiagonalFisherEstimator(parameter)
    gradient.estimator = estimator

    estimator.capture()

    assert gradient.reentered is True
    assert estimator.observation_count == 1


def test_reciprocal_merges_are_deadlock_free_and_linearizable():
    a_parameter = Tensor([0.0], requires_grad=True)
    b_parameter = Tensor([0.0], requires_grad=True)
    a_parameter.grad = np.array([1.0])
    b_parameter.grad = np.array([3.0])
    a = DiagonalFisherEstimator(a_parameter).capture()
    b = DiagonalFisherEstimator(b_parameter).capture()
    barrier = threading.Barrier(2)
    errors = []

    def merge(target, source):
        try:
            barrier.wait(timeout=5)
            target.merge(source)
        except BaseException as exc:  # pragma: no cover - failure diagnostic
            errors.append(exc)

    first = threading.Thread(target=merge, args=(a, b))
    second = threading.Thread(target=merge, args=(b, a))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sorted((a.observation_count, b.observation_count)) == [2, 3]
    assert sorted((a.total_weight, b.total_weight)) == [2.0, 3.0]

    a_value = float(a.diagonals()[0][0])
    b_value = float(b.diagonals()[0][0])
    # Legal serial order 1: a=(1+9)/2=5, then b=(9+2*5)/3=19/3.
    # Legal serial order 2 is symmetric: b=5, then a=(1+2*5)/3=11/3.
    assert (
        np.isclose(a_value, 5.0) and np.isclose(b_value, 19.0 / 3.0)
    ) or (
        np.isclose(b_value, 5.0) and np.isclose(a_value, 11.0 / 3.0)
    )
