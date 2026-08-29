import json

import numpy as np
import pytest

from engine.directional_curvature import directional_curvature
from engine.grad_mode import is_grad_enabled
from engine.ops import sum as tensor_sum
from engine.tensor import Tensor


def test_quadratic_scalar_curvature_matches_exact_directional_second_derivative():
    parameter = Tensor(2.0, requires_grad=True)

    report = directional_curvature(
        lambda: parameter * parameter,
        parameter,
        np.array(3.0),
        step=0.25,
    )

    # f(theta)=theta^2 has Hessian 2, so d^T H d = 2 * 3^2 = 18.
    assert report["curvature"] == pytest.approx(18.0)
    assert report["curvature_overflow"] is False
    assert report["curvature_underflow"] is False
    assert report["curvature_sign"] == 1
    assert report["baseline_loss"] == pytest.approx(4.0)
    assert report["plus_loss"] == pytest.approx(2.75**2)
    assert report["minus_loss"] == pytest.approx(1.25**2)
    assert parameter.data.item() == 2.0


def test_multi_parameter_quadratic_uses_joint_direction():
    left = Tensor([1.0, -2.0], requires_grad=True)
    right = Tensor(3.0, requires_grad=True)

    def loss():
        return tensor_sum(left * left) + 2.0 * right * right

    report = directional_curvature(
        loss,
        [left, right],
        [np.array([2.0, -1.0]), np.array(4.0)],
        step=0.125,
    )

    # diag Hessian = [2, 2, 4].
    expected = 2 * 4 + 2 * 1 + 4 * 16
    assert report["curvature"] == pytest.approx(expected)
    np.testing.assert_array_equal(left.data, [1.0, -2.0])
    assert right.data.item() == 3.0


def test_direction_scale_is_part_of_reported_curvature():
    parameter = Tensor(1.5, requires_grad=True)
    unit = directional_curvature(
        lambda: parameter * parameter,
        parameter,
        np.array(1.0),
        step=0.1,
    )
    doubled = directional_curvature(
        lambda: parameter * parameter,
        parameter,
        np.array(2.0),
        step=0.1,
    )
    assert unit["curvature"] == pytest.approx(2.0)
    assert doubled["curvature"] == pytest.approx(8.0)


def test_loss_callback_runs_under_no_grad_and_may_return_real_scalar():
    parameter = Tensor([2.0], requires_grad=True)
    observed_modes = []

    def loss():
        observed_modes.append(is_grad_enabled())
        return float(parameter.data[0] ** 2)

    report = directional_curvature(loss, parameter, np.array([1.0]), step=0.5)
    assert observed_modes == [False, False, False]
    assert report["curvature"] == pytest.approx(2.0)


def test_callback_may_return_zero_dimensional_numpy_array():
    parameter = Tensor([2.0], requires_grad=True)
    report = directional_curvature(
        lambda: np.array(parameter.data[0] ** 2),
        parameter,
        np.array([1.0]),
        step=0.5,
    )
    assert report["curvature"] == pytest.approx(2.0)


def test_global_numpy_rng_is_replayed_for_all_three_evaluations_and_restored():
    parameter = Tensor([1.0], requires_grad=True)
    np.random.seed(123456)
    before = np.random.get_state()
    draws = []

    def loss():
        draw = float(np.random.random())
        draws.append(draw)
        return parameter.data[0] ** 2 + draw

    report = directional_curvature(loss, parameter, np.array([1.0]), step=0.1)
    after = np.random.get_state()

    assert draws[0] == draws[1] == draws[2]
    assert report["curvature"] == pytest.approx(2.0)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_existing_gradient_binding_and_values_are_preserved():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    gradient = np.array([7.0, -3.0])
    parameter.grad = gradient
    version_before = parameter._version

    directional_curvature(
        lambda: tensor_sum(parameter * parameter),
        parameter,
        np.array([1.0, 0.0]),
        step=0.25,
    )

    assert parameter.grad is gradient
    np.testing.assert_array_equal(parameter.grad, [7.0, -3.0])
    assert parameter._version > version_before


def test_probe_restores_values_but_intentionally_invalidates_preexisting_graph():
    parameter = Tensor([2.0], requires_grad=True)
    output = tensor_sum(parameter * parameter)
    version_before = parameter._version

    directional_curvature(
        lambda: tensor_sum(parameter * parameter),
        parameter,
        np.array([1.0]),
        step=0.25,
    )

    np.testing.assert_array_equal(parameter.data, [2.0])
    assert parameter._version > version_before
    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()


def test_report_is_strict_json_safe():
    parameter = Tensor([1.0], requires_grad=True)
    report = directional_curvature(
        lambda: tensor_sum(parameter * parameter),
        parameter,
        np.array([1.0]),
        step=0.25,
    )
    json.dumps(report, allow_nan=False)
    assert report["method"] == "central_finite_difference"
    assert report["parameter_count"] == 1
    assert report["changed_parameter_count"] == 1
    assert report["changed_perturbation_elements"] == 2


def test_zero_curvature_is_reported_exactly_for_linear_loss():
    parameter = Tensor([2.0], requires_grad=True)
    report = directional_curvature(
        lambda: 3.0 * parameter.data[0] + 1.0,
        parameter,
        np.array([2.0]),
        step=0.125,
    )
    assert report["curvature"] == 0.0
    assert report["curvature_sign"] == 0
    assert report["curvature_overflow"] is False
    assert report["curvature_underflow"] is False


def test_float64_extreme_direction_can_use_representable_small_step():
    parameter = Tensor([0.0], requires_grad=True)
    maximum = np.finfo(np.float64).max
    with np.errstate(all="raise"):
        report = directional_curvature(
            lambda: parameter.data[0] / maximum,
            parameter,
            np.array([maximum]),
            step=0.5,
        )
    assert report["curvature"] == 0.0
    np.testing.assert_array_equal(parameter.data, [0.0])


def test_frozen_leaf_can_be_probed_without_changing_trainability():
    parameter = Tensor([2.0], requires_grad=False)
    report = directional_curvature(
        lambda: parameter.data[0] ** 2,
        parameter,
        np.array([1.0]),
        step=0.5,
    )
    assert report["curvature"] == pytest.approx(2.0)
    assert parameter.requires_grad is False


def test_empty_tensor_can_coexist_with_nonzero_direction_on_another_parameter():
    empty = Tensor(np.empty((0,)), requires_grad=True)
    scalar = Tensor(2.0, requires_grad=True)
    report = directional_curvature(
        lambda: scalar.data.item() ** 2,
        [empty, scalar],
        [np.empty((0,)), np.array(1.0)],
        step=0.5,
    )
    assert report["curvature"] == pytest.approx(2.0)
    assert report["parameter_count"] == 2
