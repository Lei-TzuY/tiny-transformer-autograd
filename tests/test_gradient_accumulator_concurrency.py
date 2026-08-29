import threading

import numpy as np

from engine.gradient_accumulator import GradientAccumulator
from engine.tensor import Tensor


def test_operations_wait_for_same_accumulator_lock():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [2.0]
    accumulator = GradientAccumulator(parameter)
    started = threading.Event()
    finished = threading.Event()

    def worker():
        started.set()
        accumulator.accumulate()
        finished.set()

    with accumulator._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(timeout=2.0)
        assert not finished.wait(timeout=0.05)

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert finished.is_set()
    np.testing.assert_array_equal(accumulator.average_gradients()[0], [2.0])


def test_same_thread_reentrant_lock_allows_public_operations():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad[...] = [3.0]
    accumulator = GradientAccumulator(parameter)

    with accumulator._lock:
        accumulator.accumulate(weight=2.0)
        state = accumulator.state_dict()
        accumulator.copy_to_grads()

    assert state["accumulation_count"] == 1
    assert state["total_weight"] == 2.0
    np.testing.assert_array_equal(parameter.grad, [3.0])
