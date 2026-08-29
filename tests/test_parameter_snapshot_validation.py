import numpy as np
import pytest

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_parameter_collection_validation_rejects_non_iterable_non_tensor_and_duplicates():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        ParameterSnapshot(123)
    with pytest.raises(TypeError, match="parameter 0"):
        ParameterSnapshot([object()])

    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSnapshot([p, p])


def test_parameter_generator_is_materialized_once():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    events = []

    def parameters():
        events.append("start")
        yield first
        yield second
        events.append("end")

    snapshot = ParameterSnapshot(parameters())
    assert events == ["start", "end"]
    assert snapshot.parameters == (first, second)


def test_explicit_values_validate_count_type_shape_dtype_and_finiteness():
    p = Tensor([1.0, 2.0], requires_grad=True)

    with pytest.raises(TypeError, match="iterable"):
        ParameterSnapshot(p, values=object())
    with pytest.raises(ValueError, match="count"):
        ParameterSnapshot([p], values=[])
    with pytest.raises(TypeError, match="NumPy array"):
        ParameterSnapshot(p, values=[[1.0, 2.0]])
    with pytest.raises(ValueError, match="shape"):
        ParameterSnapshot(p, values=np.array([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="floating dtype"):
        ParameterSnapshot(p, values=np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        ParameterSnapshot(p, values=np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        ParameterSnapshot(p, values=np.array([np.inf, 1.0]))


def test_float32_explicit_values_normalize_to_independent_float64():
    p = Tensor([0.0, 0.0], requires_grad=True)
    source = np.array([1.5, -2.5], dtype=np.float32)
    snapshot = ParameterSnapshot(p, values=source)
    source[...] = 0.0

    stored = snapshot.values()[0]
    assert stored.dtype == np.float64
    np.testing.assert_array_equal(stored, [1.5, -2.5])


def test_extended_precision_values_outside_float64_are_rejected_when_available():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble is not wider than float64")

    p = Tensor([0.0], requires_grad=True)
    value = np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    with pytest.raises(ValueError, match="fit float64"):
        ParameterSnapshot(p, values=value)


def test_capture_rejects_shape_drift_without_replacing_existing_snapshot():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    p.data = np.array([[7.0, 8.0]])

    with pytest.raises(ValueError, match="shape changed"):
        snapshot.capture()

    np.testing.assert_array_equal(snapshot.values()[0], [1.0, 2.0])


def test_restore_rejects_shape_drift_before_any_earlier_write():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )
    first.data[...] = 7.0
    second.data = np.array([[8.0]])
    first_version = first._version

    with pytest.raises(ValueError, match="parameter 1 shape changed"):
        snapshot.restore()

    np.testing.assert_array_equal(first.data, [7.0])
    assert first._version == first_version


def test_restore_preflights_late_read_only_destination_before_any_write():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0]), np.array([20.0])]
    )
    second.data.flags.writeable = False
    first_version = first._version

    with pytest.raises(ValueError, match="parameter 1 data must be writable"):
        snapshot.restore()

    np.testing.assert_array_equal(first.data, [1.0])
    assert first._version == first_version


def test_read_only_destination_is_allowed_when_restore_is_exact_noop():
    p = Tensor([1.0, 2.0], requires_grad=True)
    snapshot = ParameterSnapshot(p)
    p.data.flags.writeable = False
    version = p._version

    assert snapshot.restore() == 0
    assert p._version == version


def test_exact_shared_storage_is_rejected_before_model_writes():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    shared = first.data
    second._data = shared
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([5.0, 6.0]), np.array([7.0, 8.0])]
    )
    before = first.data.copy()
    version = first._version

    with pytest.raises(ValueError, match="storage must not overlap"):
        snapshot.restore()

    np.testing.assert_array_equal(first.data, before)
    assert first._version == version


def test_partial_overlapping_views_are_rejected_before_model_writes():
    backing = np.arange(6.0)
    first = Tensor([0.0, 1.0, 2.0], requires_grad=True)
    second = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    first._data = backing[:3]
    second._data = backing[2:5]
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0, 11.0, 12.0]), np.array([20.0, 21.0, 22.0])]
    )

    with pytest.raises(ValueError, match="storage must not overlap"):
        snapshot.restore()
    np.testing.assert_array_equal(backing, np.arange(6.0))


def test_disjoint_views_into_same_backing_allocation_are_allowed():
    backing = np.arange(6.0)
    first = Tensor([0.0, 1.0], requires_grad=True)
    second = Tensor([4.0, 5.0], requires_grad=True)
    first._data = backing[:2]
    second._data = backing[4:]
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([10.0, 11.0]), np.array([20.0, 21.0])]
    )

    assert snapshot.restore() == 2
    np.testing.assert_array_equal(backing, [10.0, 11.0, 2.0, 3.0, 20.0, 21.0])


def test_installed_alias_rejection_happens_before_body_entry():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    second._data = first.data
    snapshot = ParameterSnapshot(
        [first, second], values=[np.array([3.0]), np.array([4.0])]
    )
    entered = False

    with pytest.raises(ValueError, match="storage must not overlap"):
        with snapshot.installed():
            entered = True

    assert entered is False


def test_restore_does_not_modify_numpy_global_rng():
    p = Tensor([1.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([2.0]))
    before = np.random.get_state()

    snapshot.restore()

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
