import threading
import time

import numpy as np

from engine.pcgrad import PCGradBuffer
from engine.tensor import Tensor


def test_capture_waits_for_another_thread_holding_buffer_lock():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.asarray([1.0])
    pcgrad = PCGradBuffer(parameter)
    started = threading.Event()
    completed = threading.Event()

    def worker():
        started.set()
        pcgrad.capture()
        completed.set()

    with pcgrad._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(timeout=2.0)
        time.sleep(0.05)
        assert not completed.is_set()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert completed.is_set()
    assert pcgrad.task_count == 1


def test_same_thread_lock_is_reentrant_across_public_operations():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.asarray([1.0, -1.0])
    pcgrad = PCGradBuffer(parameter)

    with pcgrad._lock:
        assert pcgrad.capture() == 1
        tasks = pcgrad.task_gradients()
        projected = pcgrad.projected_gradients(seed=0)
        assert pcgrad.copy_to_grads(seed=0) is pcgrad
        assert pcgrad.reset() is pcgrad

    np.testing.assert_array_equal(tasks[0][0], [1.0, -1.0])
    np.testing.assert_array_equal(projected[0], [1.0, -1.0])
    assert pcgrad.task_count == 0
