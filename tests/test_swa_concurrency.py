import threading
import time

import numpy as np

from engine.swa import StochasticWeightAverage
from engine.tensor import Tensor


def test_update_waits_while_average_parameters_scope_holds_lock():
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [9.0]

    entered = threading.Event()
    release = threading.Event()
    updated = threading.Event()

    def holder():
        with swa.average_parameters():
            np.testing.assert_array_equal(p.data, [1.0])
            entered.set()
            assert release.wait(2.0)

    def updater():
        assert entered.wait(2.0)
        swa.update()
        updated.set()

    first = threading.Thread(target=holder)
    second = threading.Thread(target=updater)
    first.start()
    second.start()
    assert entered.wait(2.0)
    time.sleep(0.05)
    assert not updated.is_set()
    release.set()
    first.join(2.0)
    second.join(2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert updated.is_set()

    # The update must observe the restored entry weights (9.0), not temporary 1.0.
    np.testing.assert_array_equal(swa.averages()[0], [5.0])


def test_same_thread_nested_average_parameter_scopes_are_reentrant():
    p = Tensor([2.0])
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [8.0]

    with swa.average_parameters():
        np.testing.assert_array_equal(p.data, [2.0])
        with swa.average_parameters():
            np.testing.assert_array_equal(p.data, [2.0])
        np.testing.assert_array_equal(p.data, [2.0])

    np.testing.assert_array_equal(p.data, [8.0])


def test_state_read_waits_for_in_progress_average_scope():
    p = Tensor([3.0])
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [7.0]
    entered = threading.Event()
    release = threading.Event()
    read_done = threading.Event()
    observed = []

    def holder():
        with swa.average_parameters():
            entered.set()
            assert release.wait(2.0)

    def reader():
        assert entered.wait(2.0)
        observed.append(swa.state_dict())
        read_done.set()

    first = threading.Thread(target=holder)
    second = threading.Thread(target=reader)
    first.start()
    second.start()
    assert entered.wait(2.0)
    time.sleep(0.05)
    assert not read_done.is_set()
    release.set()
    first.join(2.0)
    second.join(2.0)
    assert read_done.is_set()
    assert observed[0]["num_averaged"] == 1
    np.testing.assert_array_equal(observed[0]["averages"][0], [3.0])
