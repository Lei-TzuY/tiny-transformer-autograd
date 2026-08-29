import json

import numpy as np
import pytest

from engine.swa import StochasticWeightAverage
from engine.tensor import Tensor


def _populated():
    p = Tensor([1.0, 3.0])
    swa = StochasticWeightAverage(p)
    swa.update()
    p.data[...] = [5.0, 7.0]
    swa.update()
    return p, swa


def test_state_round_trip_and_continuation_match_uninterrupted_average():
    p, source = _populated()
    restored = StochasticWeightAverage(p)
    restored.load_state_dict(source.state_dict())

    p.data[...] = [9.0, -1.0]
    source.update()
    restored.update()

    assert restored.num_averaged == source.num_averaged
    np.testing.assert_array_equal(restored.averages()[0], source.averages()[0])


def test_state_dict_returns_independent_arrays():
    _, swa = _populated()
    state = swa.state_dict()
    state["averages"][0][0] = 999.0
    np.testing.assert_array_equal(swa.averages()[0], [3.0, 5.0])


def test_empty_state_round_trip():
    p = Tensor([1.0])
    source = StochasticWeightAverage(p)
    restored = StochasticWeightAverage(p)
    restored.update()
    restored.load_state_dict(source.state_dict())
    assert restored.num_averaged == 0
    with pytest.raises(RuntimeError, match="no averaged checkpoints"):
        restored.averages()


def test_float32_state_is_normalized_to_float64():
    p = Tensor([1.0, 2.0])
    swa = StochasticWeightAverage(p)
    swa.load_state_dict(
        {
            "version": np.int64(1),
            "type": "StochasticWeightAverage",
            "num_averaged": np.int64(2),
            "averages": [np.array([3.0, 4.0], dtype=np.float32)],
            "future": "ignored",
        }
    )
    average = swa.averages()[0]
    assert average.dtype == np.float64
    np.testing.assert_array_equal(average, [3.0, 4.0])


def test_extended_precision_state_outside_float64_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64 range")
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    huge = np.array([np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)])

    with pytest.raises(ValueError, match="must fit float64"):
        swa.load_state_dict(
            {
                "version": 1,
                "type": "StochasticWeightAverage",
                "num_averaged": 1,
                "averages": [huge],
            }
        )


def test_malformed_state_rejection_is_transactional():
    _, swa = _populated()
    before = swa.state_dict()
    bad_states = [
        dict(before, version=2),
        dict(before, type="Other"),
        dict(before, num_averaged=-1),
        dict(before, averages=[]),
        dict(before, averages=[np.array([np.nan, 1.0])]),
        dict(before, averages=[np.array([[1.0, 2.0]])]),
    ]

    for bad in bad_states:
        with pytest.raises((TypeError, ValueError)):
            swa.load_state_dict(bad)
        assert swa.num_averaged == before["num_averaged"]
        np.testing.assert_array_equal(swa.averages()[0], before["averages"][0])


def test_empty_state_requires_empty_average_sequence():
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    with pytest.raises(ValueError, match="must not contain averages"):
        swa.load_state_dict(
            {
                "version": 1,
                "type": "StochasticWeightAverage",
                "num_averaged": 0,
                "averages": [np.array([1.0])],
            }
        )


def test_missing_or_invalid_state_fields_fail_explicitly():
    p = Tensor([1.0])
    swa = StochasticWeightAverage(p)
    valid = {
        "version": 1,
        "type": "StochasticWeightAverage",
        "num_averaged": 1,
        "averages": [np.array([1.0])],
    }
    for key in valid:
        malformed = dict(valid)
        malformed.pop(key)
        with pytest.raises((TypeError, ValueError)):
            swa.load_state_dict(malformed)

    with pytest.raises(TypeError, match="mapping"):
        swa.load_state_dict([])
    with pytest.raises(TypeError, match="iterable"):
        swa.load_state_dict(dict(valid, averages=123))
    with pytest.raises(TypeError, match="NumPy array"):
        swa.load_state_dict(dict(valid, averages=[[1.0]]))


def test_state_metadata_is_json_serializable_after_array_projection():
    _, swa = _populated()
    state = swa.state_dict()
    metadata = {key: value for key, value in state.items() if key != "averages"}
    json.dumps(metadata, allow_nan=False)
