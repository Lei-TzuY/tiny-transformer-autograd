"""Tests for deterministic read-only top-k accuracy metrics."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.topk_accuracy import topk_accuracy, topk_accuracy_report


def _rng_state_copy():
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_top1_and_top2_accuracy_report_known_values():
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ]
    )
    targets = np.array([0, 1, 1], dtype=np.int64)

    top1 = topk_accuracy_report(logits, targets)
    top2 = topk_accuracy_report(logits, targets, k=2)

    assert top1 == {
        "k": 1,
        "correct_count": 1,
        "scored_count": 3,
        "ignored_count": 0,
        "accuracy": 1.0 / 3.0,
    }
    assert top2 == {
        "k": 2,
        "correct_count": 3,
        "scored_count": 3,
        "ignored_count": 0,
        "accuracy": 1.0,
    }
    assert topk_accuracy(logits, targets, k=2) == 1.0


def test_higher_rank_logits_use_final_axis_as_classes():
    logits = np.array(
        [
            [[9.0, 1.0], [1.0, 9.0]],
            [[2.0, 3.0], [4.0, 0.0]],
        ]
    )
    targets = np.array([[0, 1], [0, 1]], dtype=np.int64)

    report = topk_accuracy_report(logits, targets)

    assert report["correct_count"] == 2
    assert report["scored_count"] == 4
    assert report["accuracy"] == 0.5


def test_ignore_index_excludes_rows_before_finiteness_validation():
    logits = np.array(
        [
            [4.0, 1.0],
            [np.nan, np.inf],
            [0.0, 5.0],
        ]
    )
    targets = np.array([0, -100, 1], dtype=np.int64)

    report = topk_accuracy_report(logits, targets, ignore_index=-100)

    assert report == {
        "k": 1,
        "correct_count": 2,
        "scored_count": 2,
        "ignored_count": 1,
        "accuracy": 1.0,
    }


def test_all_ignored_rows_return_none_accuracy_and_strict_json():
    logits = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    targets = np.array([-1, -1], dtype=np.int64)

    report = topk_accuracy_report(logits, targets, k=2, ignore_index=-1)

    assert report == {
        "k": 2,
        "correct_count": 0,
        "scored_count": 0,
        "ignored_count": 2,
        "accuracy": None,
    }
    assert topk_accuracy(logits, targets, k=2, ignore_index=-1) is None
    json.dumps(report, allow_nan=False)


def test_exact_ties_use_smaller_class_index_deterministically():
    logits = np.array([[3.0, 3.0, 3.0], [2.0, 2.0, 1.0]])
    targets = np.array([1, 1], dtype=np.int64)

    assert topk_accuracy(logits, targets, k=1) == 0.0
    assert topk_accuracy(logits, targets, k=2) == 1.0


def test_extreme_finite_logits_are_warning_free():
    limit = np.finfo(np.float64).max
    logits = np.array([[limit, -limit], [-limit, limit]])
    targets = np.array([0, 1], dtype=np.int64)

    with np.errstate(all="raise"):
        report = topk_accuracy_report(logits, targets)

    assert report["accuracy"] == 1.0


def test_read_only_inputs_and_rng_are_unchanged():
    logits = np.array([[3.0, 2.0], [1.0, 4.0]])
    targets = np.array([0, 1], dtype=np.int64)
    expected_logits = logits.copy()
    expected_targets = targets.copy()
    logits.flags.writeable = False
    targets.flags.writeable = False
    np.random.seed(123)
    before = _rng_state_copy()

    assert topk_accuracy(logits, targets) == 1.0

    np.testing.assert_array_equal(logits, expected_logits)
    np.testing.assert_array_equal(targets, expected_targets)
    _assert_rng_state_equal(before, _rng_state_copy())


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_k_requires_integer(value):
    logits = np.zeros((1, 2), dtype=np.float64)
    targets = np.zeros((1,), dtype=np.int64)

    with pytest.raises(TypeError, match="k must be an integer"):
        topk_accuracy_report(logits, targets, k=value)


@pytest.mark.parametrize("value", [0, -1, 3])
def test_k_must_fit_class_count(value):
    logits = np.zeros((1, 2), dtype=np.float64)
    targets = np.zeros((1,), dtype=np.int64)

    with pytest.raises(ValueError, match="k must be between 1 and the number of classes"):
        topk_accuracy_report(logits, targets, k=value)


def test_numpy_integer_k_and_ignore_index_are_supported():
    logits = np.array([[1.0, 2.0], [9.0, 0.0]])
    targets = np.array([1, -1], dtype=np.int64)

    report = topk_accuracy_report(
        logits,
        targets,
        k=np.int64(1),
        ignore_index=np.int32(-1),
    )

    assert report["accuracy"] == 1.0


@pytest.mark.parametrize("value", [True, 1.5, "-1"])
def test_ignore_index_requires_integer(value):
    logits = np.zeros((1, 2), dtype=np.float64)
    targets = np.zeros((1,), dtype=np.int64)

    with pytest.raises(TypeError, match="ignore_index must be an integer"):
        topk_accuracy_report(logits, targets, ignore_index=value)


def test_non_array_and_non_floating_logits_are_rejected():
    targets = np.array([0], dtype=np.int64)

    with pytest.raises(TypeError, match="logits must be a NumPy array"):
        topk_accuracy_report([[1.0, 2.0]], targets)
    with pytest.raises(TypeError, match="logits must have a floating dtype"):
        topk_accuracy_report(np.array([[1, 2]], dtype=np.int64), targets)


def test_scalar_and_zero_class_logits_are_rejected():
    with pytest.raises(ValueError, match="logits must have at least one dimension"):
        topk_accuracy_report(np.array(1.0), np.array(0, dtype=np.int64))
    with pytest.raises(ValueError, match="logits must have at least one class"):
        topk_accuracy_report(
            np.empty((2, 0), dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
        )


def test_target_shape_and_dtype_are_explicitly_validated():
    logits = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(TypeError, match="targets must be a NumPy array"):
        topk_accuracy_report(logits, [0, 1])
    with pytest.raises(ValueError, match="targets shape"):
        topk_accuracy_report(logits, np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(TypeError, match="targets must have an integer dtype"):
        topk_accuracy_report(logits, np.array([0.0, 1.0]))
    with pytest.raises(TypeError, match="targets must have an integer dtype"):
        topk_accuracy_report(logits, np.array([True, False]))


def test_out_of_range_scored_targets_are_rejected_but_ignored_values_are_not():
    logits = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="scored targets must be valid class indices"):
        topk_accuracy_report(logits, np.array([0, 3], dtype=np.int64))

    report = topk_accuracy_report(
        logits,
        np.array([0, 99], dtype=np.int64),
        ignore_index=99,
    )
    assert report["scored_count"] == 1


def test_nonfinite_scored_logits_are_rejected():
    targets = np.array([0], dtype=np.int64)

    for bad in (np.nan, np.inf, -np.inf):
        logits = np.array([[bad, 0.0]])
        with pytest.raises(ValueError, match="scored logits must contain only finite values"):
            topk_accuracy_report(logits, targets)


def test_empty_batch_is_valid_and_reports_no_scored_rows():
    logits = np.empty((0, 4), dtype=np.float64)
    targets = np.empty((0,), dtype=np.int64)

    report = topk_accuracy_report(logits, targets, k=3)

    assert report == {
        "k": 3,
        "correct_count": 0,
        "scored_count": 0,
        "ignored_count": 0,
        "accuracy": None,
    }
