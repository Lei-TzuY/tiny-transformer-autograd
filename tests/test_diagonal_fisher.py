import json

import numpy as np
import pytest

from engine.diagonal_fisher import DiagonalFisherEstimator
from engine.tensor import Tensor


def test_single_capture_matches_elementwise_squared_gradient():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = np.array([2.0, -3.0])
    estimator = DiagonalFisherEstimator(parameter)

    estimator.capture()

    (diagonal,) = estimator.diagonals()
    np.testing.assert_allclose(diagonal, [4.0, 9.0])
    assert estimator.observation_count == 1
    assert estimator.total_weight == 1.0
    assert estimator.trace_report()["trace"] == pytest.approx(13.0)


def test_weighted_capture_computes_weighted_mean_of_gradient_squares():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)

    parameter.grad = np.array([1.0, 2.0])
    estimator.capture(weight=1.0)
    parameter.grad = np.array([3.0, 4.0])
    estimator.capture(weight=3.0)

    expected = (np.array([1.0, 4.0]) + 3.0 * np.array([9.0, 16.0])) / 4.0
    np.testing.assert_allclose(estimator.diagonals()[0], expected)
    assert estimator.total_weight == 4.0
    assert estimator.observation_count == 2


def test_grad_none_is_exact_zero_observation():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    first.grad = np.array([2.0])
    second.grad = None
    estimator = DiagonalFisherEstimator([first, second])

    estimator.capture()

    diagonals = estimator.diagonals()
    np.testing.assert_array_equal(diagonals[0], [4.0])
    np.testing.assert_array_equal(diagonals[1], [0.0])


def test_multiple_parameters_and_scalar_tensor():
    scalar = Tensor(3.0, requires_grad=True)
    vector = Tensor([4.0, 5.0], requires_grad=True)
    scalar.grad = np.array(-2.0)
    vector.grad = np.array([1.5, -0.5])
    estimator = DiagonalFisherEstimator([scalar, vector]).capture()

    scalar_diagonal, vector_diagonal = estimator.diagonals()
    assert scalar_diagonal.shape == ()
    assert float(scalar_diagonal) == pytest.approx(4.0)
    np.testing.assert_allclose(vector_diagonal, [2.25, 0.25])
    assert estimator.trace_report()["trace"] == pytest.approx(6.5)


def test_float64_max_gradient_remains_available_as_scaled_diagonal():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.array([maximum, -maximum])
    estimator = DiagonalFisherEstimator(parameter)

    with np.errstate(all="raise"):
        estimator.capture()
        scaled = estimator.scaled_diagonals()[0]
        report = estimator.trace_report()

    assert scaled["scale"] == maximum
    np.testing.assert_array_equal(scaled["diagonal"], [1.0, 1.0])
    assert report["trace"] is None
    assert report["trace_overflow"] is True
    assert report["reason"] == "overflow"
    with pytest.raises(OverflowError, match="not representable"):
        estimator.diagonals()


def test_extreme_and_zero_weighted_observations_do_not_square_raw_maximum():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    parameter.grad = np.array([maximum])
    with np.errstate(all="raise"):
        estimator.capture(weight=1.0)
        parameter.grad = np.array([0.0])
        estimator.capture(weight=3.0)
        scaled = estimator.scaled_diagonals()[0]

    # Physical Fisher is max**2 / 4; the root representation is max / 2.
    assert scaled["scale"] == pytest.approx(maximum / 2.0)
    np.testing.assert_array_equal(scaled["diagonal"], [1.0])


def test_smallest_subnormal_gradient_is_preserved_in_scaled_state():
    tiny = np.nextafter(0.0, 1.0)
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([tiny])
    estimator = DiagonalFisherEstimator(parameter)

    with np.errstate(all="raise"):
        estimator.capture()
        scaled = estimator.scaled_diagonals()[0]
        report = estimator.trace_report()

    assert scaled["scale"] == tiny
    np.testing.assert_array_equal(scaled["diagonal"], [1.0])
    assert report["trace"] == 0.0
    assert report["trace_underflow"] is True


def test_merge_matches_direct_weighted_capture_and_preserves_source():
    direct_p = Tensor([0.0, 0.0], requires_grad=True)
    left_p = Tensor([0.0, 0.0], requires_grad=True)
    right_p = Tensor([0.0, 0.0], requires_grad=True)
    direct = DiagonalFisherEstimator(direct_p)
    left = DiagonalFisherEstimator(left_p)
    right = DiagonalFisherEstimator(right_p)

    for estimator, parameter, gradient, weight in (
        (direct, direct_p, [1.0, 2.0], 2.0),
        (direct, direct_p, [3.0, 4.0], 5.0),
        (left, left_p, [1.0, 2.0], 2.0),
        (right, right_p, [3.0, 4.0], 5.0),
    ):
        parameter.grad = np.array(gradient)
        estimator.capture(weight=weight)

    source_before = right.state_dict()
    left.merge(right)

    np.testing.assert_allclose(left.diagonals()[0], direct.diagonals()[0])
    assert left.total_weight == direct.total_weight == 7.0
    assert left.observation_count == direct.observation_count == 2
    source_after = right.state_dict()
    assert source_after["total_weight"] == source_before["total_weight"]
    assert source_after["observation_count"] == source_before["observation_count"]
    np.testing.assert_array_equal(
        source_after["states"][0]["diagonal"], source_before["states"][0]["diagonal"]
    )


def test_self_merge_doubles_metadata_without_changing_diagonal():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.array([2.0, 3.0])
    estimator = DiagonalFisherEstimator(parameter).capture(weight=1.5)
    before = estimator.diagonals()[0]

    estimator.merge(estimator)

    np.testing.assert_allclose(estimator.diagonals()[0], before)
    assert estimator.total_weight == 3.0
    assert estimator.observation_count == 2


def test_empty_source_merge_is_noop():
    p = Tensor([0.0], requires_grad=True)
    q = Tensor([0.0], requires_grad=True)
    p.grad = np.array([2.0])
    target = DiagonalFisherEstimator(p).capture()
    empty = DiagonalFisherEstimator(q)
    before = target.state_dict()

    target.merge(empty)

    after = target.state_dict()
    assert after["total_weight"] == before["total_weight"]
    assert after["observation_count"] == before["observation_count"]
    np.testing.assert_array_equal(after["states"][0]["diagonal"], before["states"][0]["diagonal"])


def test_reset_returns_to_no_observation_state():
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([2.0])
    estimator = DiagonalFisherEstimator(parameter).capture().reset()

    assert estimator.observation_count == 0
    assert estimator.total_weight == 0.0
    assert estimator.trace_report()["reason"] == "no_observations"
    with pytest.raises(RuntimeError, match="no observations"):
        estimator.diagonals()
    with pytest.raises(RuntimeError, match="no observations"):
        estimator.scaled_diagonals()


def test_trace_report_is_strict_json_safe_in_empty_finite_and_overflow_states():
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    json.dumps(estimator.trace_report(), allow_nan=False)

    parameter.grad = np.array([2.0])
    estimator.capture()
    json.dumps(estimator.trace_report(), allow_nan=False)

    parameter.grad = np.array([np.finfo(np.float64).max])
    estimator.reset().capture()
    json.dumps(estimator.trace_report(), allow_nan=False)


def test_capture_is_model_gradient_version_and_rng_neutral():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    gradient = np.array([1.0, -2.0])
    parameter.grad = gradient
    estimator = DiagonalFisherEstimator(parameter)
    data_before = parameter.data.copy()
    version_before = parameter._version
    rng_before = np.random.get_state()

    estimator.capture()

    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [1.0, -2.0])
    assert parameter._version == version_before
    rng_after = np.random.get_state()
    assert rng_after[0] == rng_before[0]
    np.testing.assert_array_equal(rng_after[1], rng_before[1])
    assert rng_after[2:] == rng_before[2:]


def test_exports_are_independent_copies():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    parameter.grad = np.array([1.0, 2.0])
    estimator = DiagonalFisherEstimator(parameter).capture()

    scaled = estimator.scaled_diagonals()
    diagonal = estimator.diagonals()
    scaled[0]["diagonal"][...] = 99.0
    diagonal[0][...] = 77.0

    np.testing.assert_allclose(estimator.diagonals()[0], [1.0, 4.0])
