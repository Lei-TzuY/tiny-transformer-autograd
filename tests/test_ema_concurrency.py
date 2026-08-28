import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.ema import ExponentialMovingAverage
from engine.tensor import Tensor


def test_overlapping_average_parameter_scopes_are_serialized():
    parameter = Tensor([10.0])
    ema = ExponentialMovingAverage(parameter, decay=0.0)
    parameter.data[...] = 2.0
    ema.update()
    parameter.data[...] = 10.0

    first_entered = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    observations = []

    def first_worker():
        with ema.average_parameters():
            observations.append(("first", float(parameter.data[0])))
            first_entered.set()
            assert release_first.wait(timeout=5.0)

    def second_worker():
        assert first_entered.wait(timeout=5.0)
        second_attempted.set()
        with ema.average_parameters():
            observations.append(("second", float(parameter.data[0])))
            second_entered.set()

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()

    assert first_entered.wait(timeout=5.0)
    assert second_attempted.wait(timeout=5.0)
    assert not second_entered.wait(timeout=0.1)

    release_first.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert observations == [("first", 2.0), ("second", 2.0)]
    assert np.array_equal(parameter.data, [10.0])


def test_update_waits_for_average_scope_and_reads_restored_parameters():
    parameter = Tensor([10.0])
    ema = ExponentialMovingAverage(parameter, decay=0.0)
    parameter.data[...] = 2.0
    ema.update()
    parameter.data[...] = 10.0

    scope_entered = threading.Event()
    update_attempted = threading.Event()
    update_finished = threading.Event()
    release_scope = threading.Event()

    def scope_worker():
        with ema.average_parameters():
            assert np.array_equal(parameter.data, [2.0])
            scope_entered.set()
            assert release_scope.wait(timeout=5.0)

    def update_worker():
        assert scope_entered.wait(timeout=5.0)
        update_attempted.set()
        ema.update()
        update_finished.set()

    scope = threading.Thread(target=scope_worker)
    updater = threading.Thread(target=update_worker)
    scope.start()
    updater.start()

    assert scope_entered.wait(timeout=5.0)
    assert update_attempted.wait(timeout=5.0)
    assert not update_finished.wait(timeout=0.1)

    release_scope.set()
    scope.join(timeout=5.0)
    updater.join(timeout=5.0)

    assert not scope.is_alive()
    assert not updater.is_alive()
    assert update_finished.is_set()
    assert np.array_equal(parameter.data, [10.0])
    assert np.array_equal(ema.averages()[0], [10.0])


def test_same_thread_nested_average_scopes_remain_reentrant():
    parameter = Tensor([10.0])
    ema = ExponentialMovingAverage(parameter, decay=0.0)
    parameter.data[...] = 2.0
    ema.update()
    parameter.data[...] = 10.0

    with ema.average_parameters():
        assert np.array_equal(parameter.data, [2.0])
        with ema.average_parameters():
            assert np.array_equal(parameter.data, [2.0])
        assert np.array_equal(parameter.data, [2.0])

    assert np.array_equal(parameter.data, [10.0])
