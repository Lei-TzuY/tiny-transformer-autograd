"""Fail-fast validation tests for optimizer step inputs."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


OPTIMIZERS = [
    pytest.param(lambda params: SGD(params, lr=0.1, momentum=0.9), id="sgd"),
    pytest.param(lambda params: Adam(params, lr=0.01), id="adam"),
    pytest.param(
        lambda params: AdamW(params, lr=0.01, weight_decay=0.1),
        id="adamw",
    ),
]


def _parameters():
    return [
        Tensor([1.0, -2.0], requires_grad=True),
        Tensor([0.5, 3.0], requires_grad=True),
    ]


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (list, tuple)):
            assert type(actual_value) is type(expected_value)
            for left, right in zip(actual_value, expected_value):
                if isinstance(right, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    assert left == right
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_nonfinite_late_gradient_leaves_all_parameters_and_state_unchanged(factory):
    params = _parameters()
    params[0].grad[:] = [0.25, -0.5]
    params[1].grad[:] = [np.nan, 1.0]
    optimizer = factory(params)
    before_params = [parameter.data.copy() for parameter in params]
    before_state = optimizer.state_dict()

    with pytest.raises(ValueError, match="parameter 1.*finite values"):
        optimizer.step()

    for parameter, expected in zip(params, before_params):
        np.testing.assert_array_equal(parameter.data, expected)
    _assert_state_equal(optimizer.state_dict(), before_state)


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_nonfinite_active_parameter_is_rejected_before_step(factory):
    params = _parameters()
    params[0].data[1] = np.inf
    params[0].grad[:] = [0.1, 0.2]
    params[1].grad = None
    optimizer = factory(params)
    before_state = optimizer.state_dict()

    with pytest.raises(ValueError, match="parameter 0.*before step"):
        optimizer.step()

    _assert_state_equal(optimizer.state_dict(), before_state)
    assert np.isinf(params[0].data[1])


@pytest.mark.parametrize(
    ("gradient", "error", "message"),
    [
        ([0.1, 0.2], TypeError, "must be a NumPy array"),
        (np.zeros((1, 2)), ValueError, "gradient shape mismatch"),
        (
            np.array([0.1 + 0.2j, 0.3 + 0.0j]),
            TypeError,
            "real numeric dtype",
        ),
        (np.array([0.1, np.inf]), ValueError, "finite values"),
    ],
)
def test_step_rejects_malformed_gradient_buffers(gradient, error, message):
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    parameter.grad = gradient
    optimizer = Adam([parameter])

    with pytest.raises(error, match=message):
        optimizer.step()

    np.testing.assert_array_equal(parameter.data, [1.0, 2.0])
    assert optimizer.t == 0


def test_inactive_parameter_is_not_validated_or_updated():
    active = Tensor([1.0], requires_grad=True)
    inactive = Tensor([np.nan], requires_grad=True)
    active.grad[:] = [0.5]
    inactive.grad = None
    optimizer = SGD([active, inactive], lr=0.1)

    optimizer.step()

    np.testing.assert_allclose(active.data, [0.95])
    assert np.isnan(inactive.data[0])
