import threading

import numpy as np

from engine.adafactor import Adafactor
from engine.tensor import Tensor


_TINY = np.nextafter(0.0, 1.0)


class BlockingArray(np.ndarray):
    def __new__(cls, values, entered, release):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.entered = entered
        obj.release = release
        obj.block_once = True
        return obj

    def __array_finalize__(self, source):
        self.entered = getattr(source, "entered", None)
        self.release = getattr(source, "release", None)
        self.block_once = getattr(source, "block_once", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.block_once:
            self.block_once = False
            self.entered.set()
            assert self.release.wait(timeout=5.0)


class ReentrantArray(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.callback = None
        obj.called = False
        return obj

    def __array_finalize__(self, source):
        self.callback = getattr(source, "callback", None)
        self.called = getattr(source, "called", False)

    def __setitem__(self, key, value):
        if not self.called and self.callback is not None:
            self.called = True
            self.callback()
        super().__setitem__(key, value)


def _optimizer(parameter):
    return Adafactor(
        parameter,
        lr=0.1,
        beta2=0.0,
        eps=_TINY,
        clip_threshold=10.0,
    )


def test_second_optimizer_step_waits_for_first_optimizer_transaction():
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()

    first_parameter = Tensor([1.0], requires_grad=True)
    first_parameter._data = BlockingArray([1.0], entered, release)
    first_parameter.grad = np.array([1.0])
    first = _optimizer(first_parameter)

    second_parameter = Tensor([2.0], requires_grad=True)
    second_parameter.grad = np.array([1.0])
    second = _optimizer(second_parameter)

    errors = []

    def run_first():
        try:
            first.step()
        except Exception as exc:  # pragma: no cover - failure diagnostic
            errors.append(exc)

    def run_second():
        try:
            second.step()
            second_done.set()
        except Exception as exc:  # pragma: no cover - failure diagnostic
            errors.append(exc)

    thread_a = threading.Thread(target=run_first)
    thread_b = threading.Thread(target=run_second)
    thread_a.start()
    assert entered.wait(timeout=5.0)
    thread_b.start()

    assert not second_done.wait(timeout=0.1)
    release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    assert second_done.is_set()
    np.testing.assert_allclose(first_parameter.data, [0.9])
    np.testing.assert_allclose(second_parameter.data, [1.9])


def test_state_reader_waits_for_in_progress_step():
    entered = threading.Event()
    release = threading.Event()
    reader_done = threading.Event()

    parameter = Tensor([1.0], requires_grad=True)
    parameter._data = BlockingArray([1.0], entered, release)
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)
    observed = []

    step_thread = threading.Thread(target=optimizer.step)

    def read_state():
        observed.append(optimizer.state_dict()["states"][0]["step"])
        reader_done.set()

    reader_thread = threading.Thread(target=read_state)
    step_thread.start()
    assert entered.wait(timeout=5.0)
    reader_thread.start()

    assert not reader_done.wait(timeout=0.1)
    release.set()
    step_thread.join(timeout=5.0)
    reader_thread.join(timeout=5.0)

    assert observed == [1]


def test_same_thread_state_dict_reentry_during_parameter_write_is_deadlock_free():
    parameter = Tensor([1.0], requires_grad=True)
    storage = ReentrantArray([1.0])
    parameter._data = storage
    parameter.grad = np.array([1.0])
    optimizer = _optimizer(parameter)
    observations = []
    storage.callback = lambda: observations.append(optimizer.steps)

    optimizer.step()

    assert observations == [(0,)]
    assert optimizer.steps == (1,)
    np.testing.assert_allclose(parameter.data, [0.9])


def test_zero_grad_waits_for_in_progress_step_on_another_instance():
    entered = threading.Event()
    release = threading.Event()
    zero_done = threading.Event()

    first_parameter = Tensor([1.0], requires_grad=True)
    first_parameter._data = BlockingArray([1.0], entered, release)
    first_parameter.grad = np.array([1.0])
    first = _optimizer(first_parameter)

    second_parameter = Tensor([2.0], requires_grad=True)
    second_parameter.grad = np.array([3.0])
    second = _optimizer(second_parameter)

    thread_a = threading.Thread(target=first.step)

    def clear():
        second.zero_grad()
        zero_done.set()

    thread_b = threading.Thread(target=clear)
    thread_a.start()
    assert entered.wait(timeout=5.0)
    thread_b.start()

    assert not zero_done.wait(timeout=0.1)
    release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert zero_done.is_set()
    np.testing.assert_array_equal(second_parameter.grad, [0.0])
