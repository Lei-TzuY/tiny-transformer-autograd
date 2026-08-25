"""Tests for the finite-difference autograd checker."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.gradcheck import gradcheck
from engine.tensor import Tensor
from nn.layers import Linear


def test_checks_multiple_inputs_and_a_non_scalar_vjp():
    x = Tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.8]])
    weight = Tensor([[0.5, -0.2], [0.1, 0.8], [-0.6, 0.4]])

    assert gradcheck(
        lambda a, b: ops.tanh(ops.matmul(a, b)),
        x,
        weight,
        atol=2e-6,
        rtol=2e-5,
    )


def test_checks_broadcasting_gradients():
    x = Tensor(np.arange(6.0).reshape(2, 3) / 5.0)
    scale = Tensor([[0.3, -0.7, 1.2]])

    assert gradcheck(lambda a, b: ops.sigmoid(a * b + a), x, scale)


def test_catches_an_incorrect_backward_rule():
    def broken_square(x):
        out = Tensor(
            x.data ** 2,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="broken_square",
        )

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                # Deliberately wrong: d(x^2)/dx is 2x, not 3x.
                x.grad += 3.0 * x.data * out.grad

        out._backward = _backward
        return out

    with pytest.raises(AssertionError, match=r"input 0 at index"):
        gradcheck(broken_square, Tensor([0.4, -1.2, 2.0]))


def test_checks_named_module_parameters_and_inputs_together():
    np.random.seed(4)
    layer = Linear(3, 2)
    x = Tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.8]])

    assert gradcheck(
        lambda value: ops.tanh(layer(value)),
        x,
        parameters=layer.named_parameters(),
        atol=2e-6,
        rtol=2e-5,
    )


def test_parameter_only_gradcheck_needs_no_dummy_input():
    weight = Tensor([0.2, -0.5, 1.1], requires_grad=True)

    assert gradcheck(lambda: ops.tanh(weight * weight), parameters=[("weight", weight)])


def test_catches_an_incorrect_parameter_backward_rule():
    weight = Tensor([0.7, -0.4], requires_grad=True)

    def broken_scale(x):
        out = Tensor(
            x.data * weight.data,
            requires_grad=x.requires_grad or weight.requires_grad,
            _children=(x, weight),
            _op="broken_scale",
        )

        def _backward():
            if x.requires_grad:
                x._ensure_grad()
                x.grad += weight.data * out.grad
            if weight.requires_grad:
                weight._ensure_grad()
                # Deliberately wrong: d(x*w)/dw is x, not 2*x.
                weight.grad += 2.0 * x.data * out.grad

        out._backward = _backward
        return out

    with pytest.raises(AssertionError, match=r"parameter 'weight' at index"):
        gradcheck(
            broken_scale,
            Tensor([0.3, 1.2]),
            parameters=[("weight", weight)],
        )


def test_parameter_data_and_existing_gradients_are_restored():
    weight = Tensor([[0.2, -0.4], [0.7, 1.1]], requires_grad=True)
    weight.grad[:] = 9.0
    data_before = weight.data.copy()
    grad_before = weight.grad.copy()

    assert gradcheck(
        lambda x: ops.matmul(x, weight),
        Tensor([[0.6, -0.3]]),
        parameters=[("weight", weight)],
    )

    np.testing.assert_array_equal(weight.data, data_before)
    np.testing.assert_array_equal(weight.grad, grad_before)


def test_parameter_state_is_restored_after_failure():
    weight = Tensor([0.4, -0.9], requires_grad=True)
    weight.grad[:] = 5.0
    data_before = weight.data.copy()
    grad_before = weight.grad.copy()

    def fails_during_numerical(x):
        if not x.requires_grad and weight.data[0] != data_before[0]:
            raise RuntimeError("numerical failure")
        return x * weight

    with pytest.raises(RuntimeError, match="numerical failure"):
        gradcheck(
            fails_during_numerical,
            Tensor([0.2, 0.3]),
            parameters=[("weight", weight)],
        )

    np.testing.assert_array_equal(weight.data, data_before)
    np.testing.assert_array_equal(weight.grad, grad_before)


def test_replays_randomness_and_restores_the_callers_rng_state():
    np.random.seed(17)
    state_before = np.random.get_state()

    def randomized_scale(x):
        scale = Tensor(np.random.uniform(0.5, 1.5, size=x.shape))
        return x * scale

    assert gradcheck(randomized_scale, Tensor([[0.2, -0.5], [1.0, 0.7]]))

    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_does_not_mutate_original_input_gradients():
    x = Tensor([0.2, -0.3, 0.9], requires_grad=True)
    x.grad[:] = 7.0

    assert gradcheck(lambda value: value * value, x)
    np.testing.assert_array_equal(x.grad, np.full(3, 7.0))


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: gradcheck(lambda: Tensor(1.0)), ValueError, "at least one"),
        (
            lambda: gradcheck(lambda x: x, np.ones(2)),
            TypeError,
            "input 0 must be a Tensor",
        ),
        (
            lambda: gradcheck(lambda x: x, Tensor([1.0]), eps=0.0),
            ValueError,
            "eps must be positive",
        ),
        (
            lambda: gradcheck(lambda x: x.data, Tensor([1.0])),
            TypeError,
            "must return a Tensor",
        ),
        (
            lambda: gradcheck(lambda x: x, Tensor([np.inf])),
            ValueError,
            "finite values",
        ),
        (
            lambda: gradcheck(
                lambda x: x,
                Tensor([1.0]),
                parameters=[Tensor([2.0])],
            ),
            ValueError,
            "parameter 0 must require gradients",
        ),
        (
            lambda: gradcheck(
                lambda x: x,
                Tensor([1.0]),
                parameters=[("weight", np.ones(1))],
            ),
            TypeError,
            "parameter 0 must be a Tensor",
        ),
    ],
)
def test_validates_public_arguments(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_rejects_duplicate_parameter_entries():
    parameter = Tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        gradcheck(
            lambda: parameter * parameter,
            parameters=[("first", parameter), ("second", parameter)],
        )
