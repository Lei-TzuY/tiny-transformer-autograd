import numpy as np
import pytest

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_state_dict_round_trip_restores_snapshot_without_touching_model():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor(3.0, requires_grad=False)
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0, 20.0]), np.array(30.0)]
    )
    state = snapshot.state_dict()

    restored = ParameterSnapshot([first, second])
    before_first = first.data.copy()
    before_second = second.data.copy()
    restored.load_state_dict(state)

    np.testing.assert_array_equal(restored.values()[0], [10.0, 20.0])
    assert restored.values()[1].item() == 30.0
    np.testing.assert_array_equal(first.data, before_first)
    np.testing.assert_array_equal(second.data, before_second)


def test_state_dict_returns_independent_arrays():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([5.0, 6.0]))

    state = snapshot.state_dict()
    state["values"][0][0] = 99.0

    np.testing.assert_array_equal(snapshot.values()[0], [5.0, 6.0])


def test_loaded_state_can_be_installed_and_restored_normally():
    p = Tensor([1.0], requires_grad=True)
    source = ParameterSnapshot(p, values=np.array([9.0]))
    target = ParameterSnapshot(p)
    target.load_state_dict(source.state_dict())

    assert target.restore() == 1
    np.testing.assert_array_equal(p.data, [9.0])


def test_float32_state_values_normalize_to_float64():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    state = snapshot.state_dict()
    state["values"] = [np.array([3.5, -4.5], dtype=np.float32)]

    snapshot.load_state_dict(state)

    stored = snapshot.values()[0]
    assert stored.dtype == np.float64
    np.testing.assert_array_equal(stored, [3.5, -4.5])


def test_state_accepts_unknown_forward_compatible_metadata():
    p = Tensor([1.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    state = snapshot.state_dict()
    state["future_metadata"] = {"ignored": True}
    state["values"] = [np.array([7.0])]

    assert snapshot.load_state_dict(state) is snapshot
    np.testing.assert_array_equal(snapshot.values()[0], [7.0])


def test_state_mapping_version_and_type_validation():
    p = Tensor([1.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)

    with pytest.raises(TypeError, match="state must be a mapping"):
        snapshot.load_state_dict([])

    state = snapshot.state_dict()
    for bad in (True, 1.5, "1"):
        candidate = dict(state)
        candidate["version"] = bad
        with pytest.raises(TypeError, match="non-negative integer"):
            snapshot.load_state_dict(candidate)

    candidate = dict(state)
    candidate["version"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        snapshot.load_state_dict(candidate)

    candidate = dict(state)
    candidate["version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        snapshot.load_state_dict(candidate)

    candidate = dict(state)
    candidate["type"] = "OtherSnapshot"
    with pytest.raises(ValueError, match="type must be"):
        snapshot.load_state_dict(candidate)


def test_state_values_count_shape_dtype_and_finiteness_validation():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    base = snapshot.state_dict()

    cases = [
        (None, TypeError, "iterable"),
        ([], ValueError, "count"),
        ([[1.0, 2.0]], TypeError, "NumPy array"),
        ([np.array([[1.0, 2.0]])], ValueError, "shape"),
        ([np.array([1, 2], dtype=np.int64)], TypeError, "floating dtype"),
        ([np.array([np.nan, 2.0])], ValueError, "finite"),
        ([np.array([np.inf, 2.0])], ValueError, "finite"),
    ]
    for values, error, pattern in cases:
        state = dict(base)
        state["values"] = values
        with pytest.raises(error, match=pattern):
            snapshot.load_state_dict(state)


def test_extended_precision_state_outside_float64_is_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([1.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    state = snapshot.state_dict()
    state["values"] = [
        np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    ]

    with pytest.raises(ValueError, match="fit float64"):
        snapshot.load_state_dict(state)


def test_malformed_state_is_transactional_for_snapshot_and_model():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([5.0, 6.0]))
    before_snapshot = snapshot.values()[0].copy()
    before_data = p.data.copy()
    version = p._version
    state = snapshot.state_dict()
    state["values"] = [np.array([np.nan, 9.0])]

    with pytest.raises(ValueError, match="finite"):
        snapshot.load_state_dict(state)

    np.testing.assert_array_equal(snapshot.values()[0], before_snapshot)
    np.testing.assert_array_equal(p.data, before_data)
    assert p._version == version


def test_load_state_rejects_live_parameter_shape_drift_without_snapshot_commit():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([5.0, 6.0]))
    before = snapshot.values()[0].copy()
    state = snapshot.state_dict()
    state["values"] = [np.array([7.0, 8.0])]
    p.data = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="shape changed"):
        snapshot.load_state_dict(state)

    np.testing.assert_array_equal(snapshot.values()[0], before)


def test_empty_snapshot_state_round_trip():
    snapshot = ParameterSnapshot([])
    state = snapshot.state_dict()
    assert state["values"] == []

    restored = ParameterSnapshot([])
    restored.load_state_dict(state)
    assert restored.values() == ()
