import sys

import numpy as np
import pytest

from engine.swa import StochasticWeightAverage
from engine.tensor import Tensor


class _FailOnceArray(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).copy().view(cls)
        obj.fail_next = True
        return obj

    def __array_finalize__(self, source):
        self.fail_next = getattr(source, "fail_next", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected write failure")


class _FailOnSecondWriteArray(np.ndarray):
    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.float64).copy().view(cls)
        obj.write_count = 0
        return obj

    def __array_finalize__(self, source):
        self.write_count = getattr(source, "write_count", 0)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.write_count += 1
        if self.write_count == 2:
            raise RuntimeError("injected restoration failure")


def test_constructor_rejects_invalid_collection_and_duplicates():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        StochasticWeightAverage(123)
    with pytest.raises(TypeError, match="parameter 1 must be a Tensor"):
        StochasticWeightAverage([Tensor([1.0]), object()])

    p = Tensor([1.0])
    with pytest.raises(ValueError, match="duplicate Tensor identities"):
        StochasticWeightAverage([p, p])


def test_update_rejects_shape_drift_without_advancing_state():
    p = Tensor([1.0, 2.0])
    swa = StochasticWeightAverage(p)
    swa.update()
    before = swa.state_dict()
    p.data = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="shape changed"):
        swa.update()

    assert swa.num_averaged == before["num_averaged"]
    np.testing.assert_array_equal(swa.averages()[0], before["averages"][0])


def test_late_nonfinite_parameter_rejection_is_transactional():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()
    before = swa.state_dict()
    p1.data[...] = [3.0]
    p2.data[...] = [np.nan]

    with pytest.raises(ValueError, match="parameter 1 data must be finite"):
        swa.update()

    assert swa.num_averaged == before["num_averaged"]
    for actual, expected in zip(swa.averages(), before["averages"]):
        np.testing.assert_array_equal(actual, expected)


def test_copy_preflights_late_read_only_destination_before_first_write():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()
    p1.data[...] = [10.0]
    p2.data[...] = [20.0]
    before1 = p1.data.copy()
    before2 = p2.data.copy()
    p2.data.flags.writeable = False

    with pytest.raises(ValueError, match="parameter 1 data must be writable"):
        swa.copy_to_parameters()

    np.testing.assert_array_equal(p1.data, before1)
    np.testing.assert_array_equal(p2.data, before2)


def test_mutate_then_raise_copy_failure_rolls_back_all_attempted_parameters():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()
    p1.data[...] = [10.0]
    p2._data = _FailOnceArray([20.0])
    before1 = p1.data.copy()
    before2 = np.array(p2.data, copy=True)

    with pytest.raises(RuntimeError, match="injected write failure"):
        swa.copy_to_parameters()

    np.testing.assert_array_equal(p1.data, before1)
    np.testing.assert_array_equal(p2.data, before2)


def test_copy_rollback_failure_still_restores_other_attempted_parameters():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    p3 = Tensor([3.0])
    swa = StochasticWeightAverage([p1, p2, p3])
    swa.update()

    p1.data[...] = [10.0]
    p2._data = _FailOnSecondWriteArray([20.0])
    p3._data = _FailOnceArray([30.0])
    before1 = p1.data.copy()
    before2 = np.array(p2.data, copy=True)
    before3 = np.array(p3.data, copy=True)

    with pytest.raises(RuntimeError, match="SWA parameter rollback failed"):
        swa.copy_to_parameters()

    np.testing.assert_array_equal(p1.data, before1)
    np.testing.assert_array_equal(p2.data, before2)
    np.testing.assert_array_equal(p3.data, before3)


def test_empty_state_cannot_be_copied_or_used_as_context():
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    with pytest.raises(RuntimeError, match="no averaged checkpoints"):
        swa.copy_to_parameters()
    with pytest.raises(RuntimeError, match="no averaged checkpoints"):
        with swa.average_parameters():
            pass


def test_loaded_maximum_count_rejects_next_update_before_reading_parameters():
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    swa.load_state_dict(
        {
            "version": 1,
            "type": "StochasticWeightAverage",
            "num_averaged": sys.maxsize,
            "averages": [np.array([2.0])],
        }
    )
    p.data[...] = [np.nan]

    with pytest.raises(OverflowError, match="supported maximum"):
        swa.update()


def test_average_context_entry_failure_leaves_parameters_unchanged():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()
    p1.data[...] = [5.0]
    p2.data[...] = [6.0]
    before1 = p1.data.copy()
    before2 = p2.data.copy()
    p2.data.flags.writeable = False

    with pytest.raises(ValueError, match="must be writable"):
        with swa.average_parameters():
            pass

    np.testing.assert_array_equal(p1.data, before1)
    np.testing.assert_array_equal(p2.data, before2)


def test_average_context_restoration_failure_still_restores_later_parameters():
    p1 = Tensor([1.0])
    p2 = Tensor([2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()

    p1._data = _FailOnSecondWriteArray([10.0])
    p2.data[...] = [20.0]
    before1 = np.array(p1.data, copy=True)
    before2 = p2.data.copy()

    with pytest.raises(RuntimeError, match="SWA parameter restoration failed"):
        with swa.average_parameters():
            np.testing.assert_array_equal(p1.data, [1.0])
            np.testing.assert_array_equal(p2.data, [2.0])

    np.testing.assert_array_equal(p1.data, before1)
    np.testing.assert_array_equal(p2.data, before2)
