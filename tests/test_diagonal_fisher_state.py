import copy
import json

import numpy as np
import pytest

from engine.diagonal_fisher import DiagonalFisherEstimator
from engine.tensor import Tensor


def _make_estimator():
    first = Tensor([0.0, 0.0], requires_grad=True)
    second = Tensor(0.0, requires_grad=True)
    estimator = DiagonalFisherEstimator([first, second])
    first.grad = np.array([1.0, 2.0])
    second.grad = np.array(3.0)
    estimator.capture(weight=2.0)
    first.grad = np.array([4.0, 5.0])
    second.grad = np.array(-6.0)
    estimator.capture(weight=3.0)
    return estimator, first, second


def test_state_round_trip_preserves_diagonals_and_metadata():
    estimator, _, _ = _make_estimator()
    state = estimator.state_dict()
    p = Tensor([0.0, 0.0], requires_grad=True)
    q = Tensor(0.0, requires_grad=True)
    restored = DiagonalFisherEstimator([p, q]).load_state_dict(state)

    assert restored.total_weight == estimator.total_weight
    assert restored.observation_count == estimator.observation_count
    for actual, expected in zip(restored.diagonals(), estimator.diagonals()):
        np.testing.assert_allclose(actual, expected)


def test_state_arrays_are_independent_from_estimator_and_input():
    estimator, _, _ = _make_estimator()
    state = estimator.state_dict()
    state["states"][0]["diagonal"][...] = 0.0
    assert np.any(estimator.state_dict()["states"][0]["diagonal"] != 0.0)

    clean = estimator.state_dict()
    p = Tensor([0.0, 0.0], requires_grad=True)
    q = Tensor(0.0, requires_grad=True)
    restored = DiagonalFisherEstimator([p, q]).load_state_dict(clean)
    clean["states"][0]["diagonal"][...] = 0.0
    assert np.any(restored.state_dict()["states"][0]["diagonal"] != 0.0)


def test_resume_then_capture_matches_uninterrupted_accumulation():
    baseline_p = Tensor([0.0, 0.0], requires_grad=True)
    resumed_p = Tensor([0.0, 0.0], requires_grad=True)
    baseline = DiagonalFisherEstimator(baseline_p)
    before = DiagonalFisherEstimator(resumed_p)

    for estimator, parameter in ((baseline, baseline_p), (before, resumed_p)):
        parameter.grad = np.array([1.0, 3.0])
        estimator.capture(weight=2.0)
        parameter.grad = np.array([2.0, 4.0])
        estimator.capture(weight=5.0)

    checkpoint = before.state_dict()
    restored_p = Tensor([0.0, 0.0], requires_grad=True)
    restored = DiagonalFisherEstimator(restored_p).load_state_dict(checkpoint)

    baseline_p.grad = np.array([7.0, -1.0])
    restored_p.grad = np.array([7.0, -1.0])
    baseline.capture(weight=3.0)
    restored.capture(weight=3.0)

    np.testing.assert_allclose(restored.diagonals()[0], baseline.diagonals()[0])
    assert restored.total_weight == baseline.total_weight
    assert restored.observation_count == baseline.observation_count


def test_float32_state_normalizes_to_float64():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    estimator.load_state_dict(
        {
            "version": 1,
            "type": "DiagonalFisherEstimator",
            "total_weight": np.float32(2.0),
            "observation_count": np.int64(1),
            "states": [
                {
                    "scale": np.float32(2.0),
                    "diagonal": np.array([1.0, 0.25], dtype=np.float32),
                }
            ],
        }
    )

    state = estimator.state_dict()
    assert isinstance(state["total_weight"], float)
    assert state["states"][0]["diagonal"].dtype == np.float64
    np.testing.assert_allclose(estimator.diagonals()[0], [4.0, 1.0])


def test_noncanonical_positive_state_is_canonicalized_without_changing_physical_values():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    estimator.load_state_dict(
        {
            "version": 1,
            "type": "DiagonalFisherEstimator",
            "total_weight": 1.0,
            "observation_count": 1,
            "states": [
                {"scale": 4.0, "diagonal": np.array([0.25, 0.0625])}
            ],
        }
    )

    saved = estimator.state_dict()["states"][0]
    assert saved["scale"] == pytest.approx(2.0)
    np.testing.assert_allclose(saved["diagonal"], [1.0, 0.25])
    np.testing.assert_allclose(estimator.diagonals()[0], [4.0, 1.0])


def test_overflow_scaled_state_round_trip_remains_usable():
    maximum = np.finfo(np.float64).max
    parameter = Tensor([0.0], requires_grad=True)
    parameter.grad = np.array([maximum])
    estimator = DiagonalFisherEstimator(parameter).capture()
    state = estimator.state_dict()

    restored_parameter = Tensor([0.0], requires_grad=True)
    restored = DiagonalFisherEstimator(restored_parameter).load_state_dict(state)
    assert restored.trace_report()["trace_overflow"] is True
    assert restored.scaled_diagonals()[0]["scale"] == maximum

    restored_parameter.grad = np.array([0.0])
    restored.capture(weight=3.0)
    assert restored.scaled_diagonals()[0]["scale"] == pytest.approx(maximum / 2.0)


def test_unknown_extra_metadata_is_tolerated():
    estimator, _, _ = _make_estimator()
    state = estimator.state_dict()
    state["future_metadata"] = {"anything": True}
    for item in state["states"]:
        item["future"] = "ignored"

    p = Tensor([0.0, 0.0], requires_grad=True)
    q = Tensor(0.0, requires_grad=True)
    restored = DiagonalFisherEstimator([p, q]).load_state_dict(state)
    for actual, expected in zip(restored.diagonals(), estimator.diagonals()):
        np.testing.assert_allclose(actual, expected)


def test_malformed_state_is_transactional():
    estimator, _, _ = _make_estimator()
    before = estimator.state_dict()
    bad = estimator.state_dict()
    bad["states"][1]["diagonal"] = np.array(np.nan)

    with pytest.raises(ValueError, match="finite"):
        estimator.load_state_dict(bad)

    after = estimator.state_dict()
    assert after["total_weight"] == before["total_weight"]
    assert after["observation_count"] == before["observation_count"]
    for actual, expected in zip(after["states"], before["states"]):
        assert actual["scale"] == expected["scale"]
        np.testing.assert_array_equal(actual["diagonal"], expected["diagonal"])


def test_state_envelope_validation():
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    base = {
        "version": 1,
        "type": "DiagonalFisherEstimator",
        "total_weight": 0.0,
        "observation_count": 0,
        "states": [{"scale": 0.0, "diagonal": np.array([0.0])}],
    }

    cases = []
    bad = dict(base); bad["version"] = 2; cases.append((bad, ValueError))
    bad = dict(base); bad["version"] = True; cases.append((bad, TypeError))
    bad = dict(base); bad["type"] = "Other"; cases.append((bad, ValueError))
    bad = dict(base); bad["total_weight"] = -1.0; cases.append((bad, ValueError))
    bad = dict(base); bad["total_weight"] = np.inf; cases.append((bad, ValueError))
    bad = dict(base); bad["observation_count"] = -1; cases.append((bad, ValueError))
    bad = dict(base); bad["observation_count"] = True; cases.append((bad, TypeError))
    bad = dict(base); bad["total_weight"] = 1.0; cases.append((bad, ValueError))
    bad = dict(base); bad["states"] = []; cases.append((bad, ValueError))
    bad = dict(base); bad["states"] = "bad"; cases.append((bad, TypeError))

    for bad_state, error in cases:
        with pytest.raises(error):
            estimator.load_state_dict(bad_state)


def test_per_parameter_state_validation():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    base = {
        "version": 1,
        "type": "DiagonalFisherEstimator",
        "total_weight": 1.0,
        "observation_count": 1,
        "states": [{"scale": 2.0, "diagonal": np.array([1.0, 0.25])}],
    }

    variants = []
    bad = copy.deepcopy(base); bad["states"][0]["scale"] = -1.0; variants.append((bad, ValueError))
    bad = copy.deepcopy(base); bad["states"][0]["scale"] = True; variants.append((bad, TypeError))
    bad = copy.deepcopy(base); bad["states"][0]["diagonal"] = [1.0, 0.25]; variants.append((bad, TypeError))
    bad = copy.deepcopy(base); bad["states"][0]["diagonal"] = np.array([1.0]); variants.append((bad, ValueError))
    bad = copy.deepcopy(base); bad["states"][0]["diagonal"] = np.array([1.0, -0.1]); variants.append((bad, ValueError))
    bad = copy.deepcopy(base); bad["states"][0] = {"scale": 0.0, "diagonal": np.array([1.0, 0.0])}; variants.append((bad, ValueError))
    bad = copy.deepcopy(base); bad["states"][0] = {"scale": 1.0, "diagonal": np.array([0.0, 0.0])}; variants.append((bad, ValueError))

    for bad_state, error in variants:
        with pytest.raises(error):
            estimator.load_state_dict(bad_state)


def test_wider_than_float64_saved_diagonal_is_rejected():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not exceed float64")
    parameter = Tensor([0.0], requires_grad=True)
    estimator = DiagonalFisherEstimator(parameter)
    too_large = np.array([np.finfo(np.float64).max], dtype=np.longdouble) * np.longdouble(2)

    with pytest.raises(ValueError, match="fit float64"):
        estimator.load_state_dict(
            {
                "version": 1,
                "type": "DiagonalFisherEstimator",
                "total_weight": 1.0,
                "observation_count": 1,
                "states": [{"scale": 1.0, "diagonal": too_large}],
            }
        )


def test_loading_state_never_changes_model_gradients_or_versions():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    gradient = np.array([3.0, 4.0])
    parameter.grad = gradient
    estimator = DiagonalFisherEstimator(parameter)
    version = parameter._version
    data = parameter.data.copy()

    estimator.load_state_dict(
        {
            "version": 1,
            "type": "DiagonalFisherEstimator",
            "total_weight": 1.0,
            "observation_count": 1,
            "states": [{"scale": 2.0, "diagonal": np.array([1.0, 0.25])}],
        }
    )

    np.testing.assert_array_equal(parameter.data, data)
    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [3.0, 4.0])
    assert parameter._version == version
