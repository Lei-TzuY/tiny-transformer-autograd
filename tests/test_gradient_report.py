"""Regression tests for read-only gradient health diagnostics."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradient_report import gradient_report
from engine.tensor import Tensor


def _rng_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_report_summarizes_finite_zero_and_frozen_gradients():
    first = Tensor([1.0, 2.0], requires_grad=True)
    second = Tensor([3.0, 4.0], requires_grad=True)
    frozen = Tensor([5.0], requires_grad=False)
    first.grad[...] = [3.0, 4.0]
    second.grad[...] = 0.0

    report = gradient_report([first, second, frozen])

    assert report["parameter_count"] == 3
    assert report["trainable_parameter_count"] == 2
    assert report["frozen_parameter_count"] == 1
    assert report["parameter_element_count"] == 5
    assert report["gradient_present_count"] == 2
    assert report["missing_gradient_count"] == 0
    assert report["unexpected_gradient_count"] == 0
    assert report["invalid_gradient_count"] == 0
    assert report["nonfinite_gradient_count"] == 0
    assert report["zero_gradient_count"] == 1
    assert report["anomaly_count"] == 0
    assert report["gradient_element_count"] == 4
    assert report["finite_element_count"] == 4
    assert report["zero_element_count"] == 2
    assert report["nan_element_count"] == 0
    assert report["posinf_element_count"] == 0
    assert report["neginf_element_count"] == 0
    assert report["max_finite_abs_gradient"] == 4.0
    assert report["trainable_global_l2_norm"] == 5.0
    assert report["trainable_global_l2_overflow"] is False
    assert [entry["status"] for entry in report["entries"]] == [
        "finite",
        "zero",
        "not_required",
    ]


def test_named_generator_is_materialized_once_and_names_are_preserved():
    tensors = [Tensor([1.0], requires_grad=True), Tensor([2.0], requires_grad=True)]
    yielded = []

    def items():
        for name, tensor in zip(("left", "right"), tensors):
            yielded.append(name)
            yield name, tensor

    report = gradient_report(items())

    assert yielded == ["left", "right"]
    assert [entry["name"] for entry in report["entries"]] == ["left", "right"]
    assert [entry["index"] for entry in report["entries"]] == [0, 1]


def test_direct_tensor_input_is_supported():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad[...] = [1.0, -2.0]

    report = gradient_report(tensor)

    assert report["parameter_count"] == 1
    assert report["entries"][0]["name"] is None
    assert report["entries"][0]["shape"] == [2]
    assert report["entries"][0]["gradient_dtype"] == "float64"
    assert report["entries"][0]["gradient_shape"] == [2]
    assert report["entries"][0]["l2_norm"] == pytest.approx(np.sqrt(5.0))


def test_missing_trainable_gradient_is_an_anomaly_and_disables_global_norm():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad = None

    report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "missing"
    assert entry["anomaly"] is True
    assert report["missing_gradient_count"] == 1
    assert report["anomaly_count"] == 1
    assert report["trainable_global_l2_norm"] is None
    assert report["trainable_global_l2_overflow"] is False


def test_nonfinite_gradient_reports_exact_element_kinds():
    tensor = Tensor(np.zeros(5), requires_grad=True)
    tensor.grad = np.array([0.0, np.nan, np.inf, -np.inf, -2.0])

    report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "nonfinite"
    assert entry["finite_elements"] == 2
    assert entry["zero_elements"] == 1
    assert entry["nan_elements"] == 1
    assert entry["posinf_elements"] == 1
    assert entry["neginf_elements"] == 1
    assert entry["max_finite_abs"] == 2.0
    assert entry["l2_norm"] is None
    assert report["nonfinite_gradient_count"] == 1
    assert report["nan_element_count"] == 1
    assert report["posinf_element_count"] == 1
    assert report["neginf_element_count"] == 1
    assert report["trainable_global_l2_norm"] is None


def test_invalid_gradient_type_and_shape_are_reported_without_raising():
    bad_type = Tensor([1.0, 2.0], requires_grad=True)
    bad_shape = Tensor([3.0, 4.0], requires_grad=True)
    bad_type.grad = [1.0, 2.0]
    bad_shape.grad = np.ones((1, 2), dtype=np.float64)

    report = gradient_report([bad_type, bad_shape])

    first, second = report["entries"]
    assert first["status"] == "invalid_type"
    assert first["gradient_type"] == "list"
    assert first["gradient_dtype"] is None
    assert second["status"] == "shape_mismatch"
    assert second["gradient_shape"] == [1, 2]
    assert second["gradient_elements"] == 2
    assert report["invalid_gradient_count"] == 2
    assert report["anomaly_count"] == 2
    assert report["trainable_global_l2_norm"] is None


def test_integer_ndarray_gradient_is_invalid_type():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad = np.array([1, 2], dtype=np.int64)

    entry = gradient_report([tensor])["entries"][0]

    assert entry["status"] == "invalid_type"
    assert entry["gradient_type"] == "ndarray"
    assert entry["gradient_dtype"] is None


def test_gradient_on_frozen_tensor_is_flagged_without_hiding_statistics():
    tensor = Tensor([1.0, 2.0], requires_grad=False)
    tensor.grad = np.array([3.0, 4.0])

    report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "finite"
    assert entry["unexpected_for_frozen"] is True
    assert entry["anomaly"] is True
    assert entry["l2_norm"] == 5.0
    assert report["unexpected_gradient_count"] == 1
    assert report["anomaly_count"] == 1
    assert report["trainable_global_l2_norm"] == 0.0


def test_extreme_finite_gradients_use_stable_norm_without_warnings():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad = np.array([1e308, 1e-308], dtype=np.float64)

    with np.errstate(all="raise"):
        report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "finite"
    assert entry["l2_norm"] == pytest.approx(1e308)
    assert entry["l2_overflow"] is False
    assert report["trainable_global_l2_norm"] == pytest.approx(1e308)
    assert report["trainable_global_l2_overflow"] is False


def test_unrepresentable_global_l2_is_reported_as_overflow_not_infinity():
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    tensor.grad = np.array([1.3e308, 1.3e308], dtype=np.float64)

    with np.errstate(all="raise"):
        report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "finite"
    assert entry["l2_norm"] is None
    assert entry["l2_overflow"] is True
    assert entry["anomaly"] is True
    assert report["trainable_global_l2_norm"] is None
    assert report["trainable_global_l2_overflow"] is True
    json.dumps(report, allow_nan=False)


def test_extended_precision_magnitude_overflow_remains_json_safe():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider finite range than float64")

    tensor = Tensor([1.0], requires_grad=True)
    tensor.grad = np.array([np.finfo(np.longdouble).max], dtype=np.longdouble)

    with np.errstate(all="raise"):
        report = gradient_report([tensor])

    entry = report["entries"][0]
    assert entry["status"] == "finite"
    assert entry["max_finite_abs"] is None
    assert entry["magnitude_overflow"] is True
    assert entry["l2_norm"] is None
    assert entry["l2_overflow"] is True
    assert report["max_finite_abs_gradient"] is None
    assert report["trainable_global_l2_overflow"] is True
    json.dumps(report, allow_nan=False)


def test_report_is_gradient_and_rng_neutral_and_runs_no_backward():
    first = Tensor([2.0, 3.0], requires_grad=True)
    second = Tensor([4.0, 5.0], requires_grad=True)
    first.grad[...] = [7.0, 8.0]
    second.grad[...] = [9.0, 10.0]
    first_grad = first.grad
    second_grad = second.grad
    first_before = first.grad.copy()
    second_before = second.grad.copy()
    rng_before = np.random.get_state()

    def forbidden_backward():
        raise AssertionError("gradient_report must not execute backward closures")

    first._backward_fn = forbidden_backward
    second._backward_fn = forbidden_backward

    gradient_report([first, second])

    rng_after = np.random.get_state()
    assert first.grad is first_grad
    assert second.grad is second_grad
    np.testing.assert_array_equal(first.grad, first_before)
    np.testing.assert_array_equal(second.grad, second_before)
    assert _rng_state_equal(rng_before, rng_after)


def test_empty_collection_has_zero_norm_and_no_anomalies():
    report = gradient_report([])

    assert report["parameter_count"] == 0
    assert report["trainable_parameter_count"] == 0
    assert report["gradient_present_count"] == 0
    assert report["anomaly_count"] == 0
    assert report["max_finite_abs_gradient"] is None
    assert report["trainable_global_l2_norm"] == 0.0
    assert report["trainable_global_l2_overflow"] is False
    assert report["entries"] == []
    json.dumps(report, allow_nan=False)


def test_collection_validation_rejects_duplicates_names_and_mixed_modes():
    tensor = Tensor([1.0], requires_grad=True)
    other = Tensor([2.0], requires_grad=True)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        gradient_report([tensor, tensor])
    with pytest.raises(ValueError, match="names must be unique"):
        gradient_report([("x", tensor), ("x", other)])
    with pytest.raises(TypeError, match="cannot mix named and unnamed"):
        gradient_report([tensor, ("other", other)])


def test_collection_validation_rejects_malformed_entries():
    tensor = Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="Tensor or iterable"):
        gradient_report(123)
    with pytest.raises(TypeError, match="entries must be Tensors"):
        gradient_report([object()])
    with pytest.raises(TypeError, match="names must be strings"):
        gradient_report([(1, tensor)])
    with pytest.raises(TypeError, match="must contain Tensor values"):
        gradient_report([("x", object())])
