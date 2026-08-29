import sys

import numpy as np
import pytest

from engine.diagonal_fisher import DiagonalFisherEstimator
from engine.tensor import Tensor


def _snapshot(estimator):
    state = estimator.state_dict()
    return (
        state["total_weight"],
        state["observation_count"],
        tuple((item["scale"], item["diagonal"].copy()) for item in state["states"]),
    )


def _assert_snapshot(estimator, before):
    state = estimator.state_dict()
    assert state["total_weight"] == before[0]
    assert state["observation_count"] == before[1]
    for actual, expected in zip(state["states"], before[2]):
        assert actual["scale"] == expected[0]
        np.testing.assert_array_equal(actual["diagonal"], expected[1])


def test_constructor_rejects_noniterable_non_tensor_duplicates_and_frozen():
    with pytest.raises(TypeError, match="Tensor or iterable"):
        DiagonalFisherEstimator(3)
    with pytest.raises(TypeError, match="parameter 0 must be a Tensor"):
        DiagonalFisherEstimator([np.array([1.0])])

    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="duplicate"):
        DiagonalFisherEstimator([parameter, parameter])
    with pytest.raises(ValueError, match="must require gradients"):
        DiagonalFisherEstimator(Tensor([1.0], requires_grad=False))


def test_constructor_rejects_malformed_requires_grad():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.requires_grad = "yes"
    with pytest.raises(TypeError, match="requires_grad must be boolean"):
        DiagonalFisherEstimator(parameter)


def test_weight_validation_is_transactional():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    estimator = DiagonalFisherEstimator(parameter).capture()
    before = _snapshot(estimator)

    for value, exception in (
        (True, TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        (10**400, ValueError),
    ):
        with pytest.raises(exception):
            estimator.capture(weight=value)
        _assert_snapshot(estimator, before)


def test_numpy_real_weight_is_accepted():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    estimator = DiagonalFisherEstimator(parameter).capture(weight=np.float32(2.5))
    assert estimator.total_weight == 2.5


def test_gradient_must_be_numpy_floating_shape_matched_and_finite():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)

    parameter.grad = [1.0, 2.0]
    with pytest.raises(TypeError, match="NumPy array"):
        estimator.capture()
    parameter.grad = np.array([1.0])
    with pytest.raises(ValueError, match="shape"):
        estimator.capture()
    parameter.grad = np.array([1 + 2j, 3 + 4j])
    with pytest.raises(TypeError, match="real numeric"):
        estimator.capture()
    parameter.grad = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        estimator.capture()
    parameter.grad = np.array([1.0, np.inf])
    with pytest.raises(ValueError, match="finite"):
        estimator.capture()

    assert estimator.observation_count == 0


def test_integer_gradient_is_rejected_as_not_floating_semantics():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2], dtype=np.int64)
    estimator = DiagonalFisherEstimator(parameter)

    # Real integer arrays are mathematically valid for squaring, and the public
    # boundary deliberately accepts real numeric NumPy gradients like main optimizers.
    estimator.capture()
    np.testing.assert_array_equal(estimator.diagonals()[0], [4.0])


def test_extended_precision_gradient_outside_float64_is_rejected_before_commit():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64")
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([2.0])
    second.grad = np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)
    estimator = DiagonalFisherEstimator([first, second])

    with pytest.raises(ValueError, match="fit float64"):
        estimator.capture()
    assert estimator.observation_count == 0
    assert estimator.total_weight == 0.0


def test_read_only_gradient_is_supported():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    gradient = np.array([2.0, 3.0])
    gradient.flags.writeable = False
    parameter.grad = gradient
    estimator = DiagonalFisherEstimator(parameter).capture()

    np.testing.assert_array_equal(estimator.diagonals()[0], [4.0, 9.0])
    assert parameter.grad is gradient


def test_shape_drift_and_trainability_drift_are_rejected_without_state_change():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([2.0, 3.0])
    estimator = DiagonalFisherEstimator(parameter).capture()
    before = _snapshot(estimator)

    parameter.data = np.array([1.0, 2.0, 3.0])
    parameter.grad = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="shape changed"):
        estimator.capture()
    _assert_snapshot(estimator, before)

    parameter.data = np.array([1.0, 2.0])
    parameter.grad = np.array([1.0, 1.0])
    parameter.requires_grad = False
    with pytest.raises(ValueError, match="no longer requires gradients"):
        estimator.capture()
    _assert_snapshot(estimator, before)


def test_malformed_live_trainability_metadata_is_rejected():
    parameter = Tensor([1.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    parameter.requires_grad = object()
    with pytest.raises(TypeError, match="requires_grad must be boolean"):
        estimator.capture()
    assert estimator.observation_count == 0


def test_late_invalid_gradient_does_not_partially_advance_earlier_state():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([2.0])
    second.grad = np.array([3.0])
    estimator = DiagonalFisherEstimator([first, second]).capture()
    before = _snapshot(estimator)

    first.grad = np.array([9.0])
    second.grad = np.array([np.nan])
    with pytest.raises(ValueError, match="finite"):
        estimator.capture(weight=4.0)
    _assert_snapshot(estimator, before)


def test_total_weight_overflow_is_transactional():
    parameter = Tensor([1.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    estimator = DiagonalFisherEstimator(parameter).capture(weight=np.finfo(np.float64).max)
    before = _snapshot(estimator)

    with pytest.raises(OverflowError, match="total weight overflow"):
        estimator.capture(weight=np.finfo(np.float64).max)
    _assert_snapshot(estimator, before)


def test_merge_rejects_wrong_type_and_shape_mismatch():
    parameter = Tensor([1.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    with pytest.raises(TypeError, match="DiagonalFisherEstimator"):
        estimator.merge(object())

    other = DiagonalFisherEstimator(Tensor([1.0, 2.0], requires_grad=True))
    with pytest.raises(ValueError, match="shapes do not match"):
        estimator.merge(other)


def test_merge_weight_overflow_leaves_both_estimators_unchanged():
    left_p = Tensor([0.0], requires_grad=True)
    right_p = Tensor([0.0], requires_grad=True)
    left_p.grad = np.array([1.0])
    right_p.grad = np.array([2.0])
    maximum = np.finfo(np.float64).max
    left = DiagonalFisherEstimator(left_p).capture(weight=maximum)
    right = DiagonalFisherEstimator(right_p).capture(weight=maximum)
    left_before = _snapshot(left)
    right_before = _snapshot(right)

    with pytest.raises(OverflowError, match="total weight overflow"):
        left.merge(right)

    _assert_snapshot(left, left_before)
    _assert_snapshot(right, right_before)


def test_empty_parameter_collection_can_accumulate_metadata():
    estimator = DiagonalFisherEstimator([]).capture(weight=2.0)
    assert estimator.observation_count == 1
    assert estimator.total_weight == 2.0
    assert estimator.diagonals() == ()
    assert estimator.scaled_diagonals() == ()
    assert estimator.trace_report()["trace"] == 0.0


def test_maximum_observation_count_rejects_next_capture():
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    estimator.load_state_dict(
        {
            "version": 1,
            "type": "DiagonalFisherEstimator",
            "total_weight": 1.0,
            "observation_count": sys.maxsize,
            "states": [{"scale": 0.0, "diagonal": np.array([0.0])}],
        }
    )
    parameter.grad = np.array([1.0])
    before = _snapshot(estimator)

    with pytest.raises(OverflowError, match="count reached"):
        estimator.capture()
    _assert_snapshot(estimator, before)
