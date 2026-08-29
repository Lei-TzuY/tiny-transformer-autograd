"""Regression coverage for read-only parameter value health reports."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.parameter_report import parameter_report
from engine.tensor import Tensor


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_known_unnamed_report_and_order():
    first = Tensor([3.0, 4.0], requires_grad=True)
    second = Tensor([0.0, -12.0], requires_grad=False)

    report = parameter_report([first, second])

    assert report["named"] is False
    assert report["tensor_count"] == 2
    assert report["element_count"] == 4
    assert report["trainable_tensor_count"] == 1
    assert report["finite_count"] == 4
    assert report["nonfinite_count"] == 0
    assert report["zero_count"] == 1
    assert report["max_abs_finite"] == 12.0
    assert report["l2"] == 13.0
    assert report["l2_overflow"] is False
    assert [item["name"] for item in report["parameters"]] == ["0", "1"]
    assert report["parameters"][0]["l2"] == 5.0
    assert report["parameters"][1]["l2"] == 12.0


def test_named_report_is_deterministic_by_name():
    alpha = Tensor([1.0])
    zeta = Tensor([2.0])

    report = parameter_report([("zeta", zeta), ("alpha", alpha)])

    assert report["named"] is True
    assert [item["name"] for item in report["parameters"]] == ["alpha", "zeta"]


def test_generator_is_materialized_once():
    values = [Tensor([1.0]), Tensor([2.0])]
    seen = []

    def source():
        for tensor in values:
            seen.append(id(tensor))
            yield tensor

    report = parameter_report(source())

    assert report["tensor_count"] == 2
    assert seen == [id(tensor) for tensor in values]


def test_nonfinite_values_are_diagnosed_and_json_safe():
    tensor = Tensor([np.nan, np.inf, -np.inf, 2.0, 0.0])

    report = parameter_report(tensor)
    item = report["parameters"][0]

    assert report["finite_count"] == 2
    assert report["nonfinite_count"] == 3
    assert report["nan_count"] == 1
    assert report["positive_infinity_count"] == 1
    assert report["negative_infinity_count"] == 1
    assert report["zero_count"] == 1
    assert report["max_abs_finite"] == 2.0
    assert report["l2"] is None
    assert report["l2_overflow"] is False
    assert item["min_finite"] == 0.0
    assert item["max_finite"] == 2.0
    assert item["l2"] is None
    json.dumps(report, allow_nan=False)


def test_all_nonfinite_has_no_finite_extrema():
    report = parameter_report(Tensor([np.nan, np.inf]))
    item = report["parameters"][0]

    assert report["max_abs_finite"] is None
    assert item["min_finite"] is None
    assert item["max_finite"] is None
    assert item["max_abs_finite"] is None


def test_extreme_finite_norm_is_warning_neutral():
    value = 1.3e308
    tensor = Tensor([value, -value])

    with np.errstate(all="raise"):
        report = parameter_report(tensor)

    expected = value * np.sqrt(2.0)
    assert report["l2"] == pytest.approx(expected, rel=1e-15)
    assert report["l2_overflow"] is False


def test_true_binary64_l2_overflow_is_explicit():
    maximum = np.finfo(np.float64).max

    with np.errstate(all="raise"):
        report = parameter_report(Tensor([maximum, maximum]))

    assert report["l2"] is None
    assert report["l2_overflow"] is True
    assert report["parameters"][0]["l2"] is None
    assert report["parameters"][0]["l2_overflow"] is True
    json.dumps(report, allow_nan=False)


def test_cross_tensor_l2_overflow_is_explicit():
    maximum = np.finfo(np.float64).max
    first = Tensor([maximum])
    second = Tensor([maximum])

    with np.errstate(all="raise"):
        report = parameter_report([first, second])

    assert report["parameters"][0]["l2"] == maximum
    assert report["parameters"][1]["l2"] == maximum
    assert report["l2"] is None
    assert report["l2_overflow"] is True


def test_empty_collection_and_empty_tensor_are_supported():
    empty = parameter_report([])
    assert empty == {
        "named": False,
        "tensor_count": 0,
        "element_count": 0,
        "trainable_tensor_count": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "zero_count": 0,
        "max_abs_finite": None,
        "l2": 0.0,
        "l2_overflow": False,
        "parameters": [],
    }

    report = parameter_report(Tensor(np.empty((0, 3))))
    item = report["parameters"][0]
    assert item["shape"] == [0, 3]
    assert item["element_count"] == 0
    assert item["l2"] == 0.0
    assert item["max_abs_finite"] is None


def test_report_preserves_tensor_grad_version_and_rng():
    tensor = Tensor([1.0, -2.0], requires_grad=True)
    tensor.grad[...] = [7.0, 8.0]
    grad_object = tensor.grad
    grad_before = tensor.grad.copy()
    data_before = tensor.data.copy()
    version_before = tensor._version
    np.random.seed(12345)
    rng_before = np.random.get_state()

    parameter_report(tensor)

    np.testing.assert_array_equal(tensor.data, data_before)
    assert tensor.grad is grad_object
    np.testing.assert_array_equal(tensor.grad, grad_before)
    assert tensor._version == version_before
    assert _rng_state_equal(np.random.get_state(), rng_before)


@pytest.mark.parametrize("bad", [None, 3, object()])
def test_non_iterable_input_is_rejected(bad):
    with pytest.raises(TypeError, match="Tensor or iterable"):
        parameter_report(bad)


def test_malformed_entries_are_rejected_explicitly():
    tensor = Tensor([1.0])

    with pytest.raises(TypeError, match="Tensors or"):
        parameter_report([object()])
    with pytest.raises(TypeError, match="parameter 0 must be a Tensor"):
        parameter_report([("x", object())])
    with pytest.raises(TypeError, match="parameter name 0 must be a string"):
        parameter_report([(1, tensor)])


def test_mixed_named_and_unnamed_entries_are_rejected():
    first = Tensor([1.0])
    second = Tensor([2.0])

    with pytest.raises(ValueError, match="must not mix"):
        parameter_report([first, ("second", second)])


def test_duplicate_tensor_identity_is_rejected():
    tensor = Tensor([1.0])

    with pytest.raises(ValueError, match="duplicate Tensors"):
        parameter_report([tensor, tensor])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate parameter name"):
        parameter_report([("x", Tensor([1.0])), ("x", Tensor([2.0]))])
