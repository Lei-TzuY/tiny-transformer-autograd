import threading

import numpy as np

from engine.adaptive_gradient_clip import adaptive_clip_grad_
from engine.tensor import Tensor


class BlockingMutateThenRaise(np.ndarray):
    def __new__(cls, values, entered, release):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.entered = entered
        obj.release = release
        obj.fail_once = True
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.entered = getattr(source, "entered", None)
            self.release = getattr(source, "release", None)
            self.fail_once = getattr(source, "fail_once", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.fail_once:
            self.fail_once = False
            self.entered.set()
            if not self.release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release injected failure")
            raise RuntimeError("injected blocked failure")


def test_failed_transaction_rolls_back_before_competing_successful_call_can_commit():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    first.grad = np.array([6.0, 8.0])
    entered = threading.Event()
    release = threading.Event()
    contender_started = threading.Event()
    contender_finished = threading.Event()
    second.grad = BlockingMutateThenRaise([6.0, 8.0], entered, release)
    failures = []

    def failing_worker():
        try:
            adaptive_clip_grad_([first, second], clip_factor=0.1)
        except RuntimeError as exc:
            failures.append(str(exc))

    def successful_worker():
        contender_started.set()
        adaptive_clip_grad_(first, clip_factor=0.2)
        contender_finished.set()

    failing = threading.Thread(target=failing_worker)
    failing.start()
    assert entered.wait(timeout=5.0)
    np.testing.assert_allclose(first.grad, [0.3, 0.4])

    contender = threading.Thread(target=successful_worker)
    contender.start()
    assert contender_started.wait(timeout=5.0)
    assert not contender_finished.wait(timeout=0.1)

    release.set()
    failing.join(timeout=5.0)
    contender.join(timeout=5.0)

    assert not failing.is_alive()
    assert not contender.is_alive()
    assert failures == ["injected blocked failure"]
    assert contender_finished.is_set()
    np.testing.assert_allclose(first.grad, [0.6, 0.8])
    np.testing.assert_array_equal(second.grad, [6.0, 8.0])


def test_same_thread_reentry_during_parameter_materialization_is_supported():
    outer = Tensor([3.0, 4.0], requires_grad=True)
    inner = Tensor([3.0, 4.0], requires_grad=True)
    outer.grad = np.array([6.0, 8.0])
    inner.grad = np.array([6.0, 8.0])
    events = []

    def parameters():
        events.append("before-inner")
        assert adaptive_clip_grad_(inner, clip_factor=0.2) == 1
        events.append("after-inner")
        yield outer

    assert adaptive_clip_grad_(parameters(), clip_factor=0.1) == 1
    assert events == ["before-inner", "after-inner"]
    np.testing.assert_allclose(inner.grad, [0.6, 0.8])
    np.testing.assert_allclose(outer.grad, [0.3, 0.4])
