"""Regression tests for live Tensor collection comparison."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from engine.tensor_compare import compare_tensor_collections


def _rng_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_identical_unnamed_collections_match_exactly():
    first = [
        Tensor([1.0, 2.0], requires_grad=True),
        Tensor([[3.0], [4.0]], requires_grad=False),
    ]
    second = [
        Tensor([1.0, 2.0], requires_grad=True),
        Tensor([[3.0], [4.0]], requires_grad=False),
    ]

    report = compare_tensor_collections(first, second)

    assert report["mode"] == "unnamed"
    assert report["atol"] == 0.0
    assert report["rtol"] == 0.0
    assert report["equal_nan"] is False
    assert report["left_tensor_count"] == 2
    assert report["right_tensor_count"] == 2
    assert report["entry_count"] == 2
    assert report["matching_tensor_count"] == 2
    assert report["mismatching_tensor_count"] == 0
    assert report["compared_element_count"] == 4
    assert report["mismatch_element_count"] == 0
    assert report["max_abs_diff"] == 0.0
    assert report["max_abs_diff_overflow"] is False
    assert report["max_symmetric_relative_diff"] == 0.0
    assert report["allclose"] is True
    assert [entry["status"] for entry in report["entries"]] == ["match", "match"]
    json.dumps(report, allow_nan=False)


def test_named_collections_align_by_name_not_iteration_order():
    left_a = Tensor([1.0], requires_grad=True)
    left_b = Tensor([2.0], requires_grad=False)
    right_a = Tensor([1.0], requires_grad=True)
    right_b = Tensor([2.0], requires_grad=False)

    report = compare_tensor_collections(
        [("b", left_b), ("a", left_a)],
        [("a", right_a), ("b", right_b)],
    )

    assert report["mode"] == "named"
    assert report["allclose"] is True
    assert [entry["name"] for entry in report["entries"]] == ["a", "b"]
    assert all("index" not in entry for entry in report["entries"])


def test_named_generators_are_consumed_once():
    left = [Tensor([1.0]), Tensor([2.0])]
    right = [Tensor([1.0]), Tensor([2.0])]
    left_yields = []
    right_yields = []

    def left_items():
        for name, tensor in zip(("x", "y"), left):
            left_yields.append(name)
            yield name, tensor

    def right_items():
        for name, tensor in zip(("x", "y"), right):
            right_yields.append(name)
            yield name, tensor

    report = compare_tensor_collections(left_items(), right_items())

    assert report["allclose"] is True
    assert left_yields == ["x", "y"]
    assert right_yields == ["x", "y"]


def test_unnamed_collections_remain_position_sensitive():
    left = [Tensor([1.0]), Tensor([2.0])]
    right = [Tensor([2.0]), Tensor([1.0])]

    report = compare_tensor_collections(left, right)

    assert report["allclose"] is False
    assert report["value_mismatch_tensor_count"] == 2
    assert report["mismatch_element_count"] == 2
    assert [entry["index"] for entry in report["entries"]] == [0, 1]
    assert all(entry["issues"] == ["values"] for entry in report["entries"])


def test_tolerances_control_value_matching_and_metrics_remain_observational():
    left = Tensor([1.0, 10.0], requires_grad=True)
    right = Tensor([1.05, 10.5], requires_grad=True)

    strict = compare_tensor_collections(left, right, atol=0.1)
    relaxed = compare_tensor_collections(left, right, rtol=0.1)

    assert strict["allclose"] is False
    assert strict["entries"][0]["close_elements"] == 1
    assert strict["entries"][0]["mismatch_elements"] == 1
    assert strict["mismatch_element_count"] == 1
    assert strict["max_abs_diff"] == pytest.approx(0.5)
    assert strict["max_symmetric_relative_diff"] == pytest.approx(0.05)

    assert relaxed["allclose"] is True
    assert relaxed["entries"][0]["close_elements"] == 2
    # Difference metrics describe the actual arrays, not just failed elements.
    assert relaxed["max_abs_diff"] == pytest.approx(0.5)
    assert relaxed["max_symmetric_relative_diff"] == pytest.approx(0.05)


def test_requires_grad_is_part_of_structural_identity():
    left = Tensor([1.0, 2.0], requires_grad=True)
    right = Tensor([1.0, 2.0], requires_grad=False)

    report = compare_tensor_collections(left, right)

    entry = report["entries"][0]
    assert entry["issues"] == ["requires_grad"]
    assert entry["mismatch_elements"] == 0
    assert entry["max_abs_diff"] == 0.0
    assert report["requires_grad_mismatch_count"] == 1
    assert report["value_mismatch_tensor_count"] == 0
    assert report["allclose"] is False


def test_shape_mismatch_is_structural_and_skips_elementwise_comparison():
    left = Tensor([[1.0, 2.0]], requires_grad=True)
    right = Tensor([[1.0], [2.0]], requires_grad=True)

    report = compare_tensor_collections(left, right)

    entry = report["entries"][0]
    assert entry["issues"] == ["shape"]
    assert entry["left_shape"] == [1, 2]
    assert entry["right_shape"] == [2, 1]
    assert entry["compared_elements"] == 0
    assert entry["mismatch_elements"] == 0
    assert entry["max_abs_diff"] is None
    assert report["shape_mismatch_count"] == 1
    assert report["compared_element_count"] == 0
    assert report["allclose"] is False


def test_named_missing_entries_are_deterministic_and_directional():
    left = [("a", Tensor([1.0])), ("b", Tensor([2.0]))]
    right = [("b", Tensor([2.0])), ("c", Tensor([3.0]))]

    report = compare_tensor_collections(left, right)

    assert [entry["name"] for entry in report["entries"]] == ["a", "b", "c"]
    assert report["entries"][0]["issues"] == ["missing_right"]
    assert report["entries"][1]["issues"] == []
    assert report["entries"][2]["issues"] == ["missing_left"]
    assert report["missing_left_count"] == 1
    assert report["missing_right_count"] == 1
    assert report["matching_tensor_count"] == 1
    assert report["mismatching_tensor_count"] == 2
    assert report["allclose"] is False


def test_unnamed_length_mismatch_reports_missing_positions():
    report = compare_tensor_collections(
        [Tensor([1.0])],
        [Tensor([1.0]), Tensor([2.0]), Tensor([3.0])],
    )

    assert [entry["index"] for entry in report["entries"]] == [0, 1, 2]
    assert report["entries"][1]["issues"] == ["missing_left"]
    assert report["entries"][2]["issues"] == ["missing_left"]
    assert report["missing_left_count"] == 2
    assert report["missing_right_count"] == 0


def test_empty_side_adopts_nonempty_naming_mode():
    named = [("weight", Tensor([1.0]))]

    report = compare_tensor_collections([], named)

    assert report["mode"] == "named"
    assert report["entries"] == [
        {
            "status": "mismatch",
            "issues": ["missing_left"],
            "allclose": False,
            "compared_elements": 0,
            "close_elements": 0,
            "mismatch_elements": 0,
            "finite_pair_elements": 0,
            "finite_mismatch_elements": 0,
            "nonfinite_mismatch_elements": 0,
            "max_abs_diff": None,
            "max_abs_diff_overflow": False,
            "max_symmetric_relative_diff": None,
            "name": "weight",
        }
    ]


def test_nan_semantics_are_explicit_and_json_safe():
    left = Tensor([np.nan, 1.0])
    right = Tensor([np.nan, 1.0])

    default = compare_tensor_collections(left, right)
    equal_nan = compare_tensor_collections(left, right, equal_nan=True)

    assert default["allclose"] is False
    assert default["mismatch_element_count"] == 1
    assert default["nonfinite_mismatch_element_count"] == 1
    assert default["max_abs_diff"] == 0.0
    assert default["max_symmetric_relative_diff"] == 0.0

    assert equal_nan["allclose"] is True
    assert equal_nan["mismatch_element_count"] == 0
    assert equal_nan["entries"][0]["finite_pair_elements"] == 1
    json.dumps(default, allow_nan=False)
    json.dumps(equal_nan, allow_nan=False)


def test_equal_infinities_match_and_opposite_infinities_do_not():
    left = Tensor([np.inf, -np.inf, 2.0])
    same = Tensor([np.inf, -np.inf, 2.0])
    changed = Tensor([np.inf, np.inf, 2.0])

    same_report = compare_tensor_collections(left, same)
    changed_report = compare_tensor_collections(left, changed)

    assert same_report["allclose"] is True
    assert same_report["entries"][0]["finite_pair_elements"] == 1
    assert changed_report["allclose"] is False
    assert changed_report["mismatch_element_count"] == 1
    assert changed_report["nonfinite_mismatch_element_count"] == 1


def test_finite_opposite_extremes_mark_absolute_difference_overflow():
    left = Tensor([1.3e308])
    right = Tensor([-1.3e308])

    with np.errstate(all="raise"):
        report = compare_tensor_collections(left, right)

    entry = report["entries"][0]
    assert entry["mismatch_elements"] == 1
    assert entry["finite_mismatch_elements"] == 1
    assert entry["max_abs_diff"] is None
    assert entry["max_abs_diff_overflow"] is True
    assert entry["max_symmetric_relative_diff"] == pytest.approx(2.0)
    assert report["max_abs_diff"] is None
    assert report["max_abs_diff_overflow"] is True
    assert report["max_symmetric_relative_diff"] == pytest.approx(2.0)
    json.dumps(report, allow_nan=False)


def test_one_overflowing_tensor_cannot_hide_behind_representable_differences():
    left = [Tensor([1.3e308]), Tensor([1.0])]
    right = [Tensor([-1.3e308]), Tensor([2.0])]

    report = compare_tensor_collections(left, right)

    assert report["entries"][0]["max_abs_diff_overflow"] is True
    assert report["entries"][1]["max_abs_diff"] == 1.0
    assert report["max_abs_diff"] is None
    assert report["max_abs_diff_overflow"] is True


def test_comparison_preserves_tensor_gradients_versions_and_rng_and_runs_no_backward():
    left = Tensor([2.0, 3.0], requires_grad=True)
    right = Tensor([2.0, 3.0], requires_grad=True)
    left.grad[...] = [7.0, 8.0]
    right.grad[...] = [9.0, 10.0]
    left_grad = left.grad
    right_grad = right.grad
    left_grad_before = left.grad.copy()
    right_grad_before = right.grad.copy()
    left_version = left._version
    right_version = right._version
    rng_before = np.random.get_state()

    def forbidden_backward():
        raise AssertionError("comparison must not execute backward closures")

    # Leaves normally store the shared no-op closure. Replacing the private
    # slots makes any accidental execution loud without affecting comparison.
    left._backward_fn = forbidden_backward
    right._backward_fn = forbidden_backward

    report = compare_tensor_collections(left, right)

    rng_after = np.random.get_state()
    assert report["allclose"] is True
    assert left.grad is left_grad
    assert right.grad is right_grad
    np.testing.assert_array_equal(left.grad, left_grad_before)
    np.testing.assert_array_equal(right.grad, right_grad_before)
    assert left._version == left_version
    assert right._version == right_version
    assert _rng_state_equal(rng_before, rng_after)


def test_empty_collections_match_and_are_strict_json():
    report = compare_tensor_collections([], [])

    assert report["mode"] == "unnamed"
    assert report["left_tensor_count"] == 0
    assert report["right_tensor_count"] == 0
    assert report["entry_count"] == 0
    assert report["allclose"] is True
    assert report["max_abs_diff"] is None
    assert report["max_symmetric_relative_diff"] is None
    assert report["entries"] == []
    json.dumps(report, allow_nan=False)


def test_invalid_tolerance_fails_before_consuming_collections():
    consumed = []

    def items():
        consumed.append(True)
        yield Tensor([1.0])

    with pytest.raises(TypeError, match="atol must be a non-negative finite real number"):
        compare_tensor_collections(items(), items(), atol=True)
    assert consumed == []

    with pytest.raises(ValueError, match="rtol must be finite"):
        compare_tensor_collections(items(), items(), rtol=10**400)
    assert consumed == []

    with pytest.raises(ValueError, match="atol must be non-negative"):
        compare_tensor_collections(items(), items(), atol=-1.0)
    assert consumed == []


def test_equal_nan_validation_fails_before_consuming_collections():
    consumed = []

    def items():
        consumed.append(True)
        yield Tensor([1.0])

    with pytest.raises(TypeError, match="equal_nan must be a boolean"):
        compare_tensor_collections(items(), items(), equal_nan=1)
    assert consumed == []


def test_collection_validation_rejects_ambiguous_or_malformed_inputs():
    tensor = Tensor([1.0])
    other = Tensor([2.0])

    with pytest.raises(TypeError, match="first tensor collection must be a Tensor or iterable"):
        compare_tensor_collections(123, [tensor])
    with pytest.raises(TypeError, match="entries must be Tensors"):
        compare_tensor_collections([object()], [tensor])
    with pytest.raises(TypeError, match="Tensor names must be strings"):
        compare_tensor_collections([(1, tensor)], [("x", other)])
    with pytest.raises(TypeError, match="named entries must contain Tensor values"):
        compare_tensor_collections([("x", object())], [("x", tensor)])
    with pytest.raises(TypeError, match="cannot mix named and unnamed entries"):
        compare_tensor_collections([tensor, ("x", other)], [tensor])
    with pytest.raises(ValueError, match="must not contain duplicate Tensors"):
        compare_tensor_collections([tensor, tensor], [other])
    with pytest.raises(ValueError, match="Tensor names must be unique"):
        compare_tensor_collections([("x", tensor), ("x", other)], [("x", tensor)])


def test_nonempty_named_and_unnamed_modes_cannot_be_compared():
    with pytest.raises(ValueError, match="tensor collection naming modes must match"):
        compare_tensor_collections(
            [("weight", Tensor([1.0]))],
            [Tensor([1.0])],
        )
