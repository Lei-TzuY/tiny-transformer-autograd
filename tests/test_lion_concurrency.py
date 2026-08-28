import threading

import numpy as np

from engine.lion import Lion
from engine.tensor import Tensor


def test_same_instance_step_waits_for_existing_lion_operation_lock():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lion([parameter], lr=0.1)
    entered = threading.Event()
    finished = threading.Event()
    errors = []

    def worker():
        entered.set()
        try:
            optimizer.step()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            finished.set()

    with optimizer._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert entered.wait(timeout=2.0)
        assert not finished.wait(timeout=0.05)

    assert finished.wait(timeout=2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert errors == []
    np.testing.assert_array_equal(parameter.data, [0.9])
    assert optimizer.step_count == 1


def test_same_thread_reentrant_lock_allows_optimizer_operations():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = Lion([parameter], lr=0.1)

    with optimizer._lock:
        optimizer.step()
        state = optimizer.state_dict()
        optimizer.zero_grad()
        optimizer.load_state_dict(state)

    np.testing.assert_array_equal(parameter.data, [0.9])
    np.testing.assert_array_equal(parameter.grad, [0.0])
    assert optimizer.step_count == 1
