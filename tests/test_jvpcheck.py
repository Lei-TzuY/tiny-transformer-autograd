"""Directional JVP checking should compare real derivatives without leaking state."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine import jvpcheck
from engine.tensor import Tensor


def test_jvpcheck_scalar_square_direction():
    x = Tensor([2.0, 3.0])

    assert jvpcheck(lambda value: ops.sum(value * value), x, np.array([5.0, 7.0]))


def test_jvpcheck_vector_output_direction():
    x = Tensor([1.5, -2.0, 0.25])
    tangent = np.array([2.0, -3.0, 4.0])

    assert jvpcheck(lambda value: value * value + 3.0 * value, x, tangent)


def test_jvpcheck_multiple_inputs_joint_direction():
    x = Tensor([2.0, 3.0])
    y = Tensor([5.0, 7.0])
    tangent_x = np.array([11.0, 13.0])
    tangent_y = np.array([17.0, 19.0])

    assert jvpcheck(
        lambda left, right: left * right + left,
        (x, y),
        (tangent_x, tangent_y),
    )


def test_jvpcheck_one_element_iterable_mirrors_tangent_structure():
    x = Tensor([2.0, 3.0])
    tangent = np.array([5.0, 7.0])

    assert jvpcheck(lambda value: value * value, (x,), (tangent,))


def test_jvpcheck_detects_intentionally_wrong_backward_rule():
    def bad_square(value):
        out = Tensor(
            np.asarray(value.data) ** 2,
            requires_grad=value.requires_grad,
            _children=(value,),
            _op="bad_square",
        )

        def _backward():
            if value.requires_grad:
                value._ensure_grad()
                value.grad += 3.0 * np.asarray(value.data) * out.grad

        out._backward = _backward
        return out

    x = Tensor([2.0, 3.0])

    with pytest.raises(AssertionError, match="jvpcheck failed"):
        jvpcheck(bad_square, x, np.array([1.0, -2.0]))


def test_jvpcheck_restores_caller_rng_and_replays_matching_randomness():
    np.random.seed(12345)
    state = np.random.get_state()
    x = Tensor([2.0, 3.0])
    tangent = np.array([5.0, 7.0])

    def random_scale(value):
        noise = Tensor(np.random.uniform(0.5, 1.5, size=value.shape))
        return value * noise

    assert jvpcheck(random_scale, x, tangent)
    actual_next = np.random.random()
    np.random.set_state(state)
    expected_next = np.random.random()
    assert actual_next == expected_next


def test_jvpcheck_preserves_caller_input_state():
    x = Tensor([2.0, 3.0], requires_grad=True)
    x.grad[:] = np.array([17.0, 19.0])
    grad_buffer = x.grad
    grad_before = x.grad.copy()
    data_before = x.data.copy()
    version_before = x._version

    assert jvpcheck(lambda value: value * value, x, np.array([5.0, 7.0]))

    assert x.grad is grad_buffer
    np.testing.assert_array_equal(x.grad, grad_before)
    np.testing.assert_array_equal(x.data, data_before)
    assert x._version == version_before


def test_jvpcheck_preserves_closed_over_parameter_grad_buffer():
    parameter = Tensor([5.0, 7.0], requires_grad=True)
    parameter.grad[:] = np.array([23.0, 29.0])
    grad_buffer = parameter.grad
    grad_before = parameter.grad.copy()
    x = Tensor([2.0, 3.0])

    assert jvpcheck(lambda value: value * parameter, x, np.array([11.0, 13.0]))

    assert parameter.grad is grad_buffer
    np.testing.assert_array_equal(parameter.grad, grad_before)


def test_jvpcheck_rejects_output_shape_change_during_perturbation():
    x = Tensor([0.0])

    def shape_changing(value):
        if value.data[0] > 0.0:
            return ops.concat((value, value), axis=0)
        return value

    with pytest.raises(ValueError, match="output shape changed"):
        jvpcheck(shape_changing, x, np.array([1.0]))


def test_jvpcheck_rejects_non_tensor_function_output():
    x = Tensor([2.0])

    with pytest.raises(TypeError, match="return a Tensor"):
        jvpcheck(lambda value: np.asarray(value.data), x, np.array([1.0]))


def test_jvpcheck_rejects_nonfinite_function_output():
    x = Tensor([2.0])

    with pytest.raises(ValueError, match="output must contain only finite"):
        jvpcheck(lambda value: value * Tensor([np.inf]), x, np.array([1.0]))


@pytest.mark.parametrize(
    "inputs, message",
    [
        ((), "at least one"),
        ((np.array([1.0]),), "input 0 must be a Tensor"),
        (Tensor(np.empty((0,))), "must not be empty"),
        (Tensor([np.inf]), "only finite"),
    ],
)
def test_jvpcheck_validates_inputs(inputs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        jvpcheck(lambda value: value, inputs, np.array([1.0]))


@pytest.mark.parametrize(
    "tangents, error_type, message",
    [
        (3.0, TypeError, "iterable"),
        ((np.ones(2),), ValueError, "exactly one"),
        ((np.ones(3), np.ones(2)), ValueError, "shape mismatch"),
        ((np.array([True, False]), np.ones(2)), TypeError, "real numeric"),
        ((np.array([np.nan, 0.0]), np.ones(2)), ValueError, "finite"),
    ],
)
def test_jvpcheck_validates_multi_input_tangents(tangents, error_type, message):
    x = Tensor([2.0, 3.0])
    y = Tensor([5.0, 7.0])

    with pytest.raises(error_type, match=message):
        jvpcheck(lambda left, right: left * right, (x, y), tangents)


def test_jvpcheck_rejects_unwrapped_tangent_for_iterable_input():
    x = Tensor([2.0, 3.0])

    with pytest.raises(ValueError, match="exactly one"):
        jvpcheck(lambda value: value * value, (x,), np.ones(2))


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"eps": 0.0}, ValueError, "eps must be positive"),
        ({"eps": True}, TypeError, "eps must be a real number"),
        ({"atol": -1.0}, ValueError, "atol must be non-negative"),
        ({"rtol": np.inf}, ValueError, "rtol must be finite"),
    ],
)
def test_jvpcheck_validates_tolerances(kwargs, error_type, message):
    x = Tensor([2.0])

    with pytest.raises(error_type, match=message):
        jvpcheck(lambda value: value * value, x, np.array([1.0]), **kwargs)


def test_jvpcheck_rejects_nonfinite_perturbation():
    x = Tensor([1e308])

    with pytest.raises(ValueError, match="perturbation for input 0 must remain finite"):
        jvpcheck(lambda value: value, x, np.array([1e308]), eps=2.0)
