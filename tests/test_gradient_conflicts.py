import json

import numpy as np
import pytest

from engine.gradient_conflicts import GradientConflictAnalyzer
from engine.tensor import Tensor


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_reports_known_conflict_and_orthogonal_pairs():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer([parameter])

    parameter.grad[...] = [1.0, 0.0]
    analyzer.capture("task_a")
    parameter.grad[...] = [-1.0, 0.0]
    analyzer.capture("task_b")
    parameter.grad[...] = [0.0, 2.0]
    analyzer.capture("task_c")

    report = analyzer.report()

    assert report["task_names"] == ["task_a", "task_b", "task_c"]
    assert report["cosine_similarity_matrix"] == [
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert report["conflict_matrix"] == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]
    assert report["pair_count"] == 3
    assert report["comparable_pair_count"] == 3
    assert report["conflict_pair_count"] == 1
    assert report["conflict_fraction"] == pytest.approx(1.0 / 3.0)
    assert report["mean_pair_cosine"] == pytest.approx(-1.0 / 3.0)
    assert report["min_pair_cosine"] == -1.0
    assert report["max_pair_cosine"] == 0.0
    assert [entry["status"] for entry in report["pairs"]] == [
        "conflict",
        "orthogonal",
        "orthogonal",
    ]
    assert [entry["l2_norm"] for entry in report["tasks"]] == [1.0, 1.0, 2.0]
    assert [entry["conflict_peer_count"] for entry in report["tasks"]] == [1, 1, 0]


def test_zero_gradient_task_is_explicitly_not_directionally_comparable():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)

    parameter.grad = None
    analyzer.capture("unused")
    parameter.grad = np.array([3.0, 4.0])
    analyzer.capture("active")

    report = analyzer.report()

    assert report["cosine_similarity_matrix"] == [[None, None], [None, 1.0]]
    assert report["comparable_matrix"] == [[False, False], [False, True]]
    assert report["conflict_matrix"] == [[False, False], [False, False]]
    assert report["comparable_pair_count"] == 0
    assert report["conflict_fraction"] is None
    assert report["mean_pair_cosine"] is None
    assert report["pairs"] == [
        {
            "left": "unused",
            "right": "active",
            "cosine": None,
            "status": "undefined",
            "conflict": False,
        }
    ]
    assert report["tasks"][0]["zero_gradient"] is True
    assert report["tasks"][0]["l2_norm"] == 0.0
    assert report["tasks"][1]["l2_norm"] == 5.0


def test_cosine_is_global_across_multiple_parameters():
    first = Tensor([0.0, 0.0], requires_grad=True)
    second = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer([first, second])

    first.grad[...] = [1.0, 2.0]
    second.grad[...] = [2.0]
    analyzer.capture("left")

    first.grad[...] = [2.0, 0.0]
    second.grad[...] = [-1.0]
    analyzer.capture("right")

    expected = 0.0
    assert np.dot([1.0, 2.0, 2.0], [2.0, 0.0, -1.0]) == expected
    report = analyzer.report()
    assert report["cosine_similarity_matrix"][0][1] == 0.0
    assert report["pairs"][0]["status"] == "orthogonal"


def test_task_snapshots_and_report_are_independent_from_live_gradients():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.grad[...] = [1.0, 2.0]
    analyzer.capture("task")

    snapshots = analyzer.task_gradients()
    snapshots[0][1][0][...] = 99.0
    parameter.grad[...] = [-7.0, -8.0]

    second = analyzer.task_gradients()
    np.testing.assert_array_equal(second[0][1][0], [1.0, 2.0])
    assert analyzer.report()["tasks"][0]["l2_norm"] == pytest.approx(np.sqrt(5.0))


def test_auto_names_are_deterministic_and_skip_explicit_collisions():
    parameter = Tensor([1.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)

    assert analyzer.capture("task_0") == "task_0"
    assert analyzer.capture() == "task_1"
    assert analyzer.capture() == "task_2"
    assert analyzer.report()["task_names"] == ["task_0", "task_1", "task_2"]


def test_reset_discards_only_captured_analysis_state():
    parameter = Tensor([0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    original_grad = parameter.grad
    parameter.grad[...] = [3.0]
    analyzer.capture("old")

    assert analyzer.reset() is analyzer
    assert analyzer.task_count == 0
    assert parameter.grad is original_grad
    np.testing.assert_array_equal(parameter.grad, [3.0])
    assert analyzer.capture() == "task_0"


def test_report_is_strict_json_safe_and_fully_observational():
    parameter = Tensor([4.0, -5.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    parameter.grad[...] = [2.0, -3.0]
    analyzer.capture("one")
    parameter.grad[...] = [-1.0, 7.0]
    analyzer.capture("two")

    data_before = parameter.data.copy()
    grad_before = parameter.grad.copy()
    grad_ref = parameter.grad
    version_before = parameter._version
    np.random.seed(1234)
    rng_before = np.random.get_state()

    report = analyzer.report()
    json.dumps(report, allow_nan=False)

    np.testing.assert_array_equal(parameter.data, data_before)
    assert parameter.grad is grad_ref
    np.testing.assert_array_equal(parameter.grad, grad_before)
    assert parameter._version == version_before
    assert _rng_state_equal(np.random.get_state(), rng_before)


def test_true_l2_overflow_does_not_prevent_finite_cosine_reporting():
    parameter = Tensor([0.0, 0.0], requires_grad=True)
    analyzer = GradientConflictAnalyzer(parameter)
    maximum = np.finfo(np.float64).max

    parameter.grad[...] = [maximum, maximum]
    analyzer.capture("huge_a")
    parameter.grad[...] = [maximum, maximum]
    analyzer.capture("huge_b")

    with np.errstate(all="raise"):
        report = analyzer.report()

    assert report["tasks"][0]["l2_norm"] is None
    assert report["tasks"][0]["l2_overflow"] is True
    assert report["tasks"][1]["l2_norm"] is None
    assert report["cosine_similarity_matrix"][0][1] == pytest.approx(1.0)
    json.dumps(report, allow_nan=False)


def test_empty_analyzer_has_a_complete_json_safe_report():
    analyzer = GradientConflictAnalyzer([])
    report = analyzer.report()

    assert report == {
        "task_count": 0,
        "parameter_count": 0,
        "task_names": [],
        "cosine_similarity_matrix": [],
        "comparable_matrix": [],
        "conflict_matrix": [],
        "pair_count": 0,
        "comparable_pair_count": 0,
        "conflict_pair_count": 0,
        "conflict_fraction": None,
        "mean_pair_cosine": None,
        "min_pair_cosine": None,
        "max_pair_cosine": None,
        "pairs": [],
        "tasks": [],
    }
    json.dumps(report, allow_nan=False)
