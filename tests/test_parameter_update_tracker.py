import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import SGD
from engine.parameter_updates import ParameterUpdateTracker
from engine.tensor import Tensor


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _strict_json(report):
    return json.dumps(report, sort_keys=True, allow_nan=False)


def test_unchanged_single_tensor_report_is_exact_and_json_safe():
    parameter = Tensor([3.0, 4.0], requires_grad=True)
    tracker = ParameterUpdateTracker(parameter)

    report = tracker.report()

    assert report["mode"] == "unnamed"
    assert report["tensor_count"] == 1
    assert report["baseline_element_count"] == 2
    assert report["current_element_count"] == 2
    assert report["comparable_element_count"] == 2
    assert report["changed_element_count"] == 0
    assert report["changed_tensor_count"] == 0
    assert report["unchanged_tensor_count"] == 1
    assert report["rewritten_tensor_count"] == 0
    assert report["mutated_tensor_count"] == 0
    assert report["baseline_l2"] == pytest.approx(5.0)
    assert report["current_l2"] == pytest.approx(5.0)
    assert report["update_l2"] == pytest.approx(0.0)
    assert report["max_abs_update"] == pytest.approx(0.0)
    assert report["update_to_baseline_ratio"] == pytest.approx(0.0)
    assert report["value_changed"] is False
    assert report["changed"] is False
    assert report["entries"][0]["status"] == "unchanged"
    assert report["entries"][0]["index"] == 0
    _strict_json(report)


def test_named_generator_is_materialized_once_and_sorted_by_name():
    first = Tensor([1.0])
    second = Tensor([2.0])
    pulls = []

    def entries():
        pulls.append("start")
        yield "z.weight", first
        pulls.append("middle")
        yield "a.weight", second
        pulls.append("end")

    tracker = ParameterUpdateTracker(entries())
    assert pulls == ["start", "middle", "end"]

    first.data += 1.0
    second.data += 2.0
    report = tracker.report()

    assert report["mode"] == "named"
    assert [entry["name"] for entry in report["entries"]] == [
        "a.weight",
        "z.weight",
    ]
    assert pulls == ["start", "middle", "end"]
    _strict_json(report)


def test_update_metrics_match_known_vector_change():
    parameter = Tensor([3.0, 4.0])
    tracker = ParameterUpdateTracker(parameter)

    parameter.data += np.array([0.3, -0.4])
    report = tracker.report()
    entry = report["entries"][0]

    assert entry["status"] == "updated"
    assert entry["changed_element_count"] == 2
    assert entry["changed_fraction"] == pytest.approx(1.0)
    assert entry["update_l2"] == pytest.approx(0.5)
    assert entry["max_abs_update"] == pytest.approx(0.4)
    assert entry["update_to_baseline_ratio"] == pytest.approx(0.1)
    assert report["update_l2"] == pytest.approx(0.5)
    assert report["max_abs_update"] == pytest.approx(0.4)
    assert report["update_to_baseline_ratio"] == pytest.approx(0.1)
    assert report["changed_element_count"] == 2
    assert report["changed_tensor_count"] == 1
    assert report["value_changed"] is True
    assert report["changed"] is True


def test_real_sgd_step_is_observed_without_optimizer_integration():
    parameter = Tensor([1.0, -2.0], requires_grad=True)
    parameter.grad[:] = [0.5, -1.0]
    optimizer = SGD([parameter], lr=0.1)
    tracker = ParameterUpdateTracker(parameter)

    optimizer.step()
    report = tracker.report()

    assert report["entries"][0]["status"] == "updated"
    assert report["entries"][0]["version_delta"] > 0
    assert report["update_l2"] == pytest.approx(np.sqrt(0.05**2 + 0.1**2))
    assert report["max_abs_update"] == pytest.approx(0.1)
    assert report["mutated_tensor_count"] == 1


def test_same_value_write_is_reported_as_rewritten_not_unchanged():
    parameter = Tensor([1.0, 2.0])
    tracker = ParameterUpdateTracker(parameter)
    original = parameter.data.copy()

    parameter.data[...] = original
    report = tracker.report()

    assert report["entries"][0]["status"] == "rewritten"
    assert report["entries"][0]["changed_element_count"] == 0
    assert report["rewritten_tensor_count"] == 1
    assert report["unchanged_tensor_count"] == 0
    assert report["mutated_tensor_count"] == 1
    assert report["value_changed"] is False
    assert report["changed"] is True


def test_requires_grad_only_change_is_visible_without_value_change():
    parameter = Tensor([1.0], requires_grad=True)
    tracker = ParameterUpdateTracker(parameter)

    parameter.requires_grad = False
    report = tracker.report()

    assert report["entries"][0]["status"] == "unchanged"
    assert report["entries"][0]["requires_grad_changed"] is True
    assert report["requires_grad_changed_tensor_count"] == 1
    assert report["value_changed"] is False
    assert report["changed"] is True


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_nonfinite_current_values_are_diagnosed_not_raised(bad):
    parameter = Tensor([1.0, 2.0])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data[0] = bad

    report = tracker.report()
    entry = report["entries"][0]

    assert entry["status"] == "nonfinite"
    assert entry["nonfinite_current_count"] == 1
    assert entry["changed_element_count"] == 1
    assert report["changed_element_count"] == 1
    assert report["changed_tensor_count"] == 1
    assert report["nonfinite_tensor_count"] == 1
    assert report["nonfinite_element_count"] == 1
    assert report["all_finite"] is False
    assert report["update_metrics_available"] is False
    assert report["current_l2"] is None
    assert report["update_l2"] is None
    assert report["max_abs_update"] is None
    assert report["changed"] is True
    _strict_json(report)


def test_shape_change_disables_update_metrics_but_keeps_current_norm():
    parameter = Tensor([1.0, 2.0])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data = np.array([[3.0, 4.0]])

    report = tracker.report()
    entry = report["entries"][0]

    assert entry["status"] == "shape_changed"
    assert entry["baseline_shape"] == [2]
    assert entry["current_shape"] == [1, 2]
    assert entry["nonfinite_current_count"] == 0
    assert entry["current_l2"] == pytest.approx(5.0)
    assert report["shape_changed_tensor_count"] == 1
    assert report["structurally_stable"] is False
    assert report["update_metrics_available"] is False
    assert report["current_l2"] == pytest.approx(5.0)
    assert report["update_l2"] is None
    assert report["value_changed"] is True
    _strict_json(report)


def test_opposite_extreme_values_report_element_delta_overflow_without_warning():
    parameter = Tensor([1.3e308])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data[...] = -1.3e308

    with np.errstate(all="raise"):
        report = tracker.report()

    entry = report["entries"][0]
    assert entry["status"] == "updated"
    assert entry["update_l2"] is None
    assert entry["update_l2_overflow"] is True
    assert entry["max_abs_update"] is None
    assert entry["max_abs_update_overflow"] is True
    assert report["update_l2"] is None
    assert report["update_l2_overflow"] is True
    assert report["max_abs_update"] is None
    assert report["max_abs_update_overflow"] is True
    _strict_json(report)


def test_global_update_norm_overflow_does_not_hide_representable_max_update():
    parameter = Tensor([0.0, 0.0])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data[...] = [1.3e308, 1.3e308]

    with np.errstate(all="raise"):
        report = tracker.report()

    entry = report["entries"][0]
    assert entry["max_abs_update"] == pytest.approx(1.3e308)
    assert entry["max_abs_update_overflow"] is False
    assert entry["update_l2"] is None
    assert entry["update_l2_overflow"] is True
    assert report["max_abs_update"] == pytest.approx(1.3e308)
    assert report["max_abs_update_overflow"] is False
    assert report["update_l2"] is None
    assert report["update_l2_overflow"] is True
    assert report["baseline_zero"] is True
    assert report["update_to_baseline_ratio"] is None
    _strict_json(report)


def test_baseline_norm_overflow_remains_json_safe():
    parameter = Tensor([1.3e308, 1.3e308])
    tracker = ParameterUpdateTracker(parameter)

    with np.errstate(all="raise"):
        report = tracker.report()

    assert report["baseline_l2"] is None
    assert report["baseline_l2_overflow"] is True
    assert report["current_l2"] is None
    assert report["current_l2_overflow"] is True
    assert report["update_l2"] == pytest.approx(0.0)
    assert report["update_to_baseline_ratio"] is None
    assert report["update_to_baseline_ratio_overflow"] is False
    _strict_json(report)


def test_update_ratio_overflow_is_explicit_and_json_safe():
    tiny = np.nextafter(0.0, 1.0)
    parameter = Tensor([tiny])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data[...] = 1.0

    with np.errstate(all="raise"):
        report = tracker.report()

    assert report["baseline_l2"] == tiny
    assert report["update_l2"] == pytest.approx(1.0)
    assert report["update_to_baseline_ratio"] is None
    assert report["update_to_baseline_ratio_overflow"] is True
    assert report["entries"][0]["update_to_baseline_ratio_overflow"] is True
    _strict_json(report)


def test_report_preserves_grad_buffer_versions_and_numpy_rng():
    parameter = Tensor([2.0, 3.0], requires_grad=True)
    parameter.grad[:] = [7.0, 11.0]
    tracker = ParameterUpdateTracker(parameter)
    parameter.data += [0.5, -0.25]

    grad_object = parameter.grad
    grad_values = parameter.grad.copy()
    version = parameter._version
    np.random.seed(12345)
    rng_before = np.random.get_state()

    tracker.report()

    rng_after = np.random.get_state()
    assert parameter.grad is grad_object
    assert np.array_equal(parameter.grad, grad_values)
    assert parameter._version == version
    assert _rng_state_equal(rng_before, rng_after)


def test_refresh_rebases_values_versions_and_returns_self():
    parameter = Tensor([1.0, 2.0])
    tracker = ParameterUpdateTracker(parameter)
    parameter.data += [3.0, 4.0]

    assert tracker.refresh() is tracker
    baseline = tracker.baseline_values()
    report = tracker.report()

    assert np.array_equal(baseline[0], [4.0, 6.0])
    assert report["entries"][0]["status"] == "unchanged"
    assert report["entries"][0]["version_delta"] == 0
    assert report["changed"] is False

    baseline[0][0] = 999.0
    assert tracker.baseline_values()[0][0] == pytest.approx(4.0)


def test_refresh_failure_is_transactional():
    first = Tensor([1.0])
    second = Tensor([2.0])
    tracker = ParameterUpdateTracker([first, second])

    first.data[...] = 4.0
    second.data[...] = np.inf
    with pytest.raises(ValueError, match="must contain only finite values"):
        tracker.refresh()

    second.data[...] = 2.0
    report = tracker.report()
    assert report["entries"][0]["update_l2"] == pytest.approx(3.0)
    assert report["entries"][0]["status"] == "updated"


def test_empty_collection_has_valid_zero_report():
    tracker = ParameterUpdateTracker([])
    report = tracker.report()

    assert report["mode"] == "unnamed"
    assert report["tensor_count"] == 0
    assert report["baseline_l2"] == 0.0
    assert report["current_l2"] == 0.0
    assert report["update_l2"] == 0.0
    assert report["max_abs_update"] == 0.0
    assert report["changed"] is False
    assert report["entries"] == []
    _strict_json(report)


def test_direct_named_pair_is_supported():
    parameter = Tensor([1.0])
    tracker = ParameterUpdateTracker(("weight", parameter))
    parameter.data += 1.0

    report = tracker.report()
    assert report["mode"] == "named"
    assert report["entries"][0]["name"] == "weight"
    assert report["entries"][0]["status"] == "updated"


def test_constructor_rejects_nonfinite_baseline():
    parameter = Tensor([np.inf])
    with pytest.raises(ValueError, match="baseline 0.*finite"):
        ParameterUpdateTracker(parameter)


def test_collection_validation_is_explicit():
    parameter = Tensor([1.0])

    with pytest.raises(TypeError, match="expects a Tensor or iterable"):
        ParameterUpdateTracker(123)
    with pytest.raises(TypeError, match="cannot mix named and unnamed"):
        ParameterUpdateTracker([("x", parameter), Tensor([2.0])])
    with pytest.raises(TypeError, match="name 0 must be a string"):
        ParameterUpdateTracker([(1, parameter)])
    with pytest.raises(TypeError, match="entry 0 must contain a Tensor"):
        ParameterUpdateTracker([("x", object())])
    with pytest.raises(TypeError, match="entry 1 must be a Tensor"):
        ParameterUpdateTracker([parameter, object()])
    with pytest.raises(ValueError, match="duplicate parameter update name"):
        ParameterUpdateTracker([("x", parameter), ("x", Tensor([2.0]))])
    with pytest.raises(ValueError, match="cannot bind duplicate Tensors"):
        ParameterUpdateTracker([parameter, parameter])
    with pytest.raises(ValueError, match="cannot bind duplicate Tensors"):
        ParameterUpdateTracker([("x", parameter), ("y", parameter)])
