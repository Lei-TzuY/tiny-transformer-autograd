import threading
import time

import numpy as np

from engine.gradient_noise_scale import GradientNoiseScaleEstimator
from engine.tensor import Tensor


class BlockingGradTensor(Tensor):
    def __init__(self, data, requires_grad=True):
        self._grad_value = None
        self.block_reads = False
        self.read_entered = threading.Event()
        self.read_release = threading.Event()
        self.reenter_estimator = None
        self.reentrant_observation = None
        super().__init__(data, requires_grad=requires_grad)

    @property
    def grad(self):
        if self.reenter_estimator is not None:
            self.reentrant_observation = self.reenter_estimator.sample_count
        if self.block_reads:
            self.read_entered.set()
            if not self.read_release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release gradient read")
        return self._grad_value

    @grad.setter
    def grad(self, value):
        self._grad_value = value


def test_report_waits_for_in_progress_capture_and_sees_complete_sample():
    p = BlockingGradTensor([0.0])
    estimator = GradientNoiseScaleEstimator(p, batch_size=1)
    p.grad = np.array([3.0])
    p.block_reads = True

    capture_errors = []
    report_errors = []
    report_result = []
    report_done = threading.Event()

    def capture():
        try:
            estimator.capture()
        except BaseException as exc:  # pragma: no cover - diagnostic path
            capture_errors.append(exc)

    def report():
        try:
            report_result.append(estimator.report())
        except BaseException as exc:  # pragma: no cover - diagnostic path
            report_errors.append(exc)
        finally:
            report_done.set()

    capture_thread = threading.Thread(target=capture)
    capture_thread.start()
    assert p.read_entered.wait(timeout=2.0)

    report_thread = threading.Thread(target=report)
    report_thread.start()
    time.sleep(0.05)
    assert not report_done.is_set()

    p.read_release.set()
    capture_thread.join(timeout=2.0)
    report_thread.join(timeout=2.0)

    assert not capture_thread.is_alive()
    assert not report_thread.is_alive()
    assert capture_errors == []
    assert report_errors == []
    assert report_result[0]["sample_count"] == 1
    assert report_result[0]["mean_gradient_l2"] == 3.0


def test_gradient_getter_can_reenter_sample_count_on_same_thread():
    p = BlockingGradTensor([0.0])
    estimator = GradientNoiseScaleEstimator(p, batch_size=2)
    p.grad = np.array([4.0])
    p.reenter_estimator = estimator

    errors = []

    def capture():
        try:
            estimator.capture()
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    thread = threading.Thread(target=capture)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert p.reentrant_observation == 0
    assert estimator.sample_count == 1
