import threading
import time

import numpy as np

from engine.gradient_conflicts import GradientConflictAnalyzer
from engine.tensor import Tensor


class BlockingGradTensor(Tensor):
    def __init__(self, data):
        self._grad_value = None
        self.block_reads = False
        self.reenter = False
        self.analyzer = None
        self.entered = threading.Event()
        self.release = threading.Event()
        super().__init__(data, requires_grad=True)

    @property
    def grad(self):
        if self.reenter and self.analyzer is not None:
            _ = self.analyzer.task_count
        if self.block_reads:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test gradient getter timed out")
        return self._grad_value

    @grad.setter
    def grad(self, value):
        self._grad_value = value


def test_report_waits_for_in_progress_capture_and_sees_committed_task():
    parameter = BlockingGradTensor([0.0])
    parameter.grad[...] = [2.0]
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.block_reads = True

    capture_errors = []
    report_errors = []
    report_result = []

    def capture_worker():
        try:
            analyzer.capture("task")
        except BaseException as exc:
            capture_errors.append(exc)

    def report_worker():
        try:
            report_result.append(analyzer.report())
        except BaseException as exc:
            report_errors.append(exc)

    capture_thread = threading.Thread(target=capture_worker)
    capture_thread.start()
    assert parameter.entered.wait(timeout=5)

    report_thread = threading.Thread(target=report_worker)
    report_thread.start()
    time.sleep(0.05)
    assert report_thread.is_alive()
    assert report_result == []

    parameter.block_reads = False
    parameter.release.set()
    capture_thread.join(timeout=5)
    report_thread.join(timeout=5)

    assert not capture_thread.is_alive()
    assert not report_thread.is_alive()
    assert capture_errors == []
    assert report_errors == []
    assert report_result[0]["task_names"] == ["task"]
    assert report_result[0]["tasks"][0]["l2_norm"] == 2.0


def test_gradient_getter_can_reenter_same_analyzer_without_deadlock():
    parameter = BlockingGradTensor([0.0, 0.0])
    parameter.grad = np.array([3.0, 4.0])
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.analyzer = analyzer
    parameter.reenter = True

    assert analyzer.capture("task") == "task"
    report = analyzer.report()
    assert analyzer.task_count == 1
    assert report["tasks"][0]["l2_norm"] == 5.0
