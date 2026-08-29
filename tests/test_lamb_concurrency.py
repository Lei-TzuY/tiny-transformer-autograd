import threading

import numpy as np
import pytest

from engine.lamb import LAMB
from engine.tensor import Tensor


class BlockingMutateThenRaise(np.ndarray):
    def __new__(cls, values, mutated, release):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.mutated = mutated
        obj.release = release
        obj.calls = 0
        return obj

    def __array_finalize__(self, source):
        self.mutated = getattr(source, "mutated", None)
        self.release = getattr(source, "release", None)
        self.calls = getattr(source, "calls", 0)

    def __setitem__(self, key, value):
        self.calls += 1
        super().__setitem__(key, value)
        if self.calls == 1:
            self.mutated.set()
            assert self.release.wait(timeout=5.0)
            raise RuntimeError("blocked write failure")


class ReentrantWrite(np.ndarray):
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


def test_failed_transaction_rolls_back_before_competing_optimizer_can_commit():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])

    mutated = threading.Event()
    release = threading.Event()
    second._data = BlockingMutateThenRaise([3.0], mutated, release)

    failing = LAMB(
        [first, second],
        lr=0.05,
        betas=(0.0, 0.0),
        eps=np.nextafter(0.0, 1.0),
        weight_decay=0.0,
    )
    succeeding = LAMB(
        first,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=np.nextafter(0.0, 1.0),
        weight_decay=0.0,
    )

    failures = []
    contender_done = threading.Event()

    def run_failing():
        try:
            failing.step()
        except BaseException as exc:
            failures.append(exc)

    def run_contender():
        succeeding.step()
        contender_done.set()

    failing_thread = threading.Thread(target=run_failing)
    contender_thread = threading.Thread(target=run_contender)
    failing_thread.start()
    assert mutated.wait(timeout=5.0)

    contender_thread.start()
    assert not contender_done.wait(timeout=0.1)

    release.set()
    failing_thread.join(timeout=5.0)
    contender_thread.join(timeout=5.0)

    assert not failing_thread.is_alive()
    assert not contender_thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "blocked write failure" in str(failures[0])
    assert contender_done.is_set()
    # The failed optimizer restores 2.0 before the contender gets the lock;
    # scalar LAMB then takes a trusted 10% step from that restored value.
    np.testing.assert_allclose(first.data, [1.8], rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(second.data, [3.0])
    assert failing.steps == (0, 0)
    assert succeeding.steps == (1,)


def test_state_reader_waits_for_in_progress_cross_instance_transaction():
    first = Tensor([2.0], requires_grad=True)
    second = Tensor([3.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([1.0])
    mutated = threading.Event()
    release = threading.Event()
    second._data = BlockingMutateThenRaise([3.0], mutated, release)
    optimizer = LAMB([first, second], lr=0.05)

    step_done = threading.Event()
    read_done = threading.Event()
    failures = []
    observed = []

    def run_step():
        try:
            optimizer.step()
        except RuntimeError as exc:
            failures.append(exc)
        finally:
            step_done.set()

    def run_reader():
        observed.append(optimizer.state_dict())
        read_done.set()

    step_thread = threading.Thread(target=run_step)
    read_thread = threading.Thread(target=run_reader)
    step_thread.start()
    assert mutated.wait(timeout=5.0)
    read_thread.start()
    assert not read_done.wait(timeout=0.1)

    release.set()
    step_thread.join(timeout=5.0)
    read_thread.join(timeout=5.0)

    assert step_done.is_set()
    assert read_done.is_set()
    assert failures
    assert observed[0]["states"][0]["step"] == 0
    assert observed[0]["states"][1]["step"] == 0


def test_same_thread_reentrant_state_read_during_parameter_write_is_safe():
    parameter = Tensor([2.0], requires_grad=True)
    parameter._data = ReentrantWrite([2.0])
    parameter.grad = np.array([1.0])
    optimizer = LAMB(
        parameter,
        lr=0.1,
        betas=(0.0, 0.0),
        eps=np.nextafter(0.0, 1.0),
        weight_decay=0.0,
    )
    observed_steps = []
    parameter._data.callback = lambda: observed_steps.append(
        optimizer.state_dict()["states"][0]["step"]
    )

    optimizer.step()

    assert observed_steps == [0]
    assert optimizer.steps == (1,)
    np.testing.assert_allclose(parameter.data, [1.8], rtol=1e-15, atol=0.0)
