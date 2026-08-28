import threading

import numpy as np
import pytest

from engine.optim import SGD
from engine.optimizer_transaction import optimizer_step_transaction
from engine.tensor import Tensor


def test_same_optimizer_transactions_are_serialized_across_threads():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SGD([parameter], lr=0.1)

    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    errors = []

    def first_worker():
        try:
            with optimizer_step_transaction(optimizer):
                first_entered.set()
                if not release_first.wait(timeout=5.0):
                    raise AssertionError("first worker release timed out")
                optimizer.step()
        except BaseException as exc:  # surface worker failures in the main test
            errors.append(exc)

    def second_worker():
        try:
            if not first_entered.wait(timeout=5.0):
                raise AssertionError("first worker never entered")
            second_attempting.set()
            with optimizer_step_transaction(optimizer):
                second_entered.set()
                optimizer.step()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    assert first_entered.wait(timeout=5.0)
    second.start()
    assert second_attempting.wait(timeout=5.0)

    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert second_entered.is_set()
    np.testing.assert_allclose(parameter.data, [0.8], rtol=0.0, atol=1e-15)


def test_second_thread_observes_restored_values_after_first_rolls_back():
    class Marker(Exception):
        pass

    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)

    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    observed = []
    errors = []

    def first_worker():
        try:
            try:
                with optimizer_step_transaction(optimizer):
                    parameter.data[...] = [9.0]
                    first_entered.set()
                    if not release_first.wait(timeout=5.0):
                        raise AssertionError("first worker release timed out")
                    raise Marker("rollback")
            except Marker:
                pass
        except BaseException as exc:
            errors.append(exc)

    def second_worker():
        try:
            if not first_entered.wait(timeout=5.0):
                raise AssertionError("first worker never entered")
            second_attempting.set()
            with optimizer_step_transaction(optimizer):
                observed.append(parameter.data.copy())
                second_entered.set()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    assert first_entered.wait(timeout=5.0)
    second.start()
    assert second_attempting.wait(timeout=5.0)

    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()

    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], [1.0])
    np.testing.assert_array_equal(parameter.data, [1.0])


def test_writability_drift_on_normal_exit_is_rolled_back():
    parameter = Tensor([1.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)

    with pytest.raises(ValueError, match="parameter writability changed at index 0"):
        with optimizer_step_transaction(optimizer):
            parameter.data.setflags(write=False)

    assert parameter.data.flags.writeable
    np.testing.assert_array_equal(parameter.data, [1.0])


def test_read_only_baseline_remains_read_only_after_failed_step():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    parameter.data.setflags(write=False)
    optimizer = SGD([parameter], lr=0.1)
    state_before = optimizer.state_dict()

    with pytest.raises(ValueError):
        with optimizer_step_transaction(optimizer):
            optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0])
    assert not parameter.data.flags.writeable
    assert optimizer.state_dict()["lr"] == state_before["lr"]
    assert optimizer.state_dict()["momentum"] == state_before["momentum"]
    assert optimizer.state_dict()["weight_decay"] == state_before["weight_decay"]
    np.testing.assert_array_equal(optimizer.state_dict()["v"][0], state_before["v"][0])
