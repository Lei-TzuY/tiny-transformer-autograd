import numpy as np
import pytest

from engine.swa import StochasticWeightAverage
from engine.tensor import Tensor


def _bind_data_view(parameter, view):
    # Tensor.data intentionally copies public assignments. Injecting a view here
    # exercises custom/internal storage without weakening the public Tensor setter.
    parameter._data = view


def test_copy_rejects_exact_parameter_storage_alias_before_first_write():
    p1 = Tensor([1.0, 2.0])
    p2 = Tensor([1.0, 2.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()

    shared = np.array([10.0, 20.0])
    _bind_data_view(p1, shared)
    _bind_data_view(p2, shared)
    before = shared.copy()

    with pytest.raises(ValueError, match="data storage must not overlap"):
        swa.copy_to_parameters()

    np.testing.assert_array_equal(shared, before)


def test_copy_rejects_partially_overlapping_views_before_first_write():
    p1 = Tensor([1.0, 2.0])
    p2 = Tensor([2.0, 3.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()

    backing = np.array([10.0, 20.0, 30.0])
    _bind_data_view(p1, backing[:2])
    _bind_data_view(p2, backing[1:])
    before = backing.copy()

    with pytest.raises(ValueError, match="parameters 0 and 1"):
        swa.copy_to_parameters()

    np.testing.assert_array_equal(backing, before)


def test_average_context_rejects_overlapping_storage_before_body():
    p1 = Tensor([1.0, 2.0])
    p2 = Tensor([2.0, 3.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()

    backing = np.array([10.0, 20.0, 30.0])
    _bind_data_view(p1, backing[:2])
    _bind_data_view(p2, backing[1:])
    before = backing.copy()
    entered = False

    with pytest.raises(ValueError, match="data storage must not overlap"):
        with swa.average_parameters():
            entered = True

    assert entered is False
    np.testing.assert_array_equal(backing, before)


def test_disjoint_views_of_one_backing_array_remain_supported():
    p1 = Tensor([1.0, 2.0])
    p2 = Tensor([3.0, 4.0])
    swa = StochasticWeightAverage([p1, p2])
    swa.update()

    backing = np.array([10.0, 20.0, 30.0, 40.0])
    _bind_data_view(p1, backing[:2])
    _bind_data_view(p2, backing[2:])

    changed = swa.copy_to_parameters()

    assert changed == 2
    np.testing.assert_array_equal(backing, [1.0, 2.0, 3.0, 4.0])
