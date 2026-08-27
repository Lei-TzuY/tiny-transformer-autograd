"""Tensor shape mutations must not leave stale autograd or optimizer state."""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


STATEFUL_OPTIMIZERS = [
    pytest.param(lambda params: SGD(params, lr=0.1, momentum=0.9), id="sgd"),
    pytest.param(lambda params: Adam(params, lr=0.01), id="adam"),
    pytest.param(
        lambda params: AdamW(params, lr=0.01, weight_decay=0.1),
        id="adamw",
    ),
]


def _set_storage_shape_without_version_hook(tensor, shape):
    """Exercise ndarray's metadata mutation path, which bypasses Tensor hooks."""
    with warnings.catch_warnings():
        # NumPy 2.5 deprecates direct ndarray.shape assignment, but the operation
        # still exists and is precisely the metadata-only mutation path covered
        # here. Keep -W error active for every warning outside this statement.
        warnings.simplefilter("ignore", DeprecationWarning)
        tensor.data.shape = shape


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (list, tuple)):
            assert type(actual_value) is type(expected_value)
            assert len(actual_value) == len(expected_value)
            for left, right in zip(actual_value, expected_value):
                if isinstance(right, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    assert left == right
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_shape_attribute_mutation_invalidates_old_graph_transactionally():
    x = Tensor([2.0, 3.0], requires_grad=True)
    output = x * x
    x.grad[:] = [7.0, 11.0]
    before = x.grad.copy()

    # ndarray.shape assignment mutates metadata without calling the tracked
    # setitem/ufunc hooks, so graph validation must also snapshot shapes.
    _set_storage_shape_without_version_hook(x, (1, 2))

    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()
    np.testing.assert_array_equal(x.grad, before)


def test_new_graph_after_shape_attribute_mutation_rebuilds_library_grad():
    x = Tensor([2.0, 3.0], requires_grad=True)
    x.grad[:] = [9.0, 13.0]
    _set_storage_shape_without_version_hook(x, (1, 2))

    output = x * x
    output.backward(np.ones((1, 2)))

    assert x.grad.shape == (1, 2)
    np.testing.assert_array_equal(x.grad, np.array([[4.0, 6.0]]))


def test_new_graph_after_data_replacement_rebuilds_library_grad():
    x = Tensor([2.0, 3.0], requires_grad=True)
    x.grad[:] = [5.0, 7.0]
    x.data = np.array([[1.0, 2.0, 4.0]])

    output = x * 3.0
    output.backward(np.ones((1, 3)))

    assert x.grad.shape == (1, 3)
    np.testing.assert_array_equal(x.grad, np.full((1, 3), 3.0))


def test_correct_shape_manual_grad_survives_shape_change():
    x = Tensor([2.0, 3.0], requires_grad=True)
    x.data = np.array([[4.0, 5.0, 6.0]])
    x.grad = np.array([[10.0, 20.0, 30.0]])

    output = x * 2.0
    output.backward(np.ones((1, 3)))

    np.testing.assert_array_equal(x.grad, np.array([[12.0, 22.0, 32.0]]))


@pytest.mark.parametrize("factory", STATEFUL_OPTIMIZERS)
def test_stale_optimizer_state_shape_fails_before_any_mutation(factory):
    first = Tensor([1.0, -2.0], requires_grad=True)
    second = Tensor([0.5, 3.0], requires_grad=True)
    optimizer = factory([first, second])

    first.grad[:] = [0.25, -0.5]
    second.data = np.array([0.5, 3.0, -1.0])
    second.zero_grad()
    second.grad[:] = [0.1, 0.2, 0.3]

    before_params = [first.data.copy(), second.data.copy()]
    before_state = optimizer.state_dict()

    with pytest.raises(ValueError, match="shape mismatch for parameter 1"):
        optimizer.step()

    np.testing.assert_array_equal(first.data, before_params[0])
    np.testing.assert_array_equal(second.data, before_params[1])
    _assert_state_equal(optimizer.state_dict(), before_state)


def test_stateless_sgd_allows_parameter_shape_change():
    parameter = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1, momentum=0.0)

    parameter.data = np.array([1.0, 2.0, 3.0])
    parameter.zero_grad()
    parameter.grad[:] = [0.5, -1.0, 2.0]
    optimizer.step()

    np.testing.assert_allclose(parameter.data, np.array([0.95, 2.1, 2.8]))
