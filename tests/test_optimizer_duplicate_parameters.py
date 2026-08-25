import numpy as np
import pytest

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
def test_optimizer_rejects_duplicate_parameter_references(optimizer_cls):
    parameter = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    parameter.grad = np.array([0.25, -0.5])
    data_before = parameter.data.copy()
    grad_before = parameter.grad.copy()

    with pytest.raises(ValueError, match="duplicate"):
        optimizer_cls([parameter, parameter])

    np.testing.assert_array_equal(parameter.data, data_before)
    np.testing.assert_array_equal(parameter.grad, grad_before)


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
def test_optimizer_rejects_duplicate_references_from_generator(optimizer_cls):
    parameter = Tensor(np.array([3.0]), requires_grad=True)
    parameters = (value for value in (parameter, parameter))

    with pytest.raises(ValueError, match="duplicate"):
        optimizer_cls(parameters)


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
def test_optimizer_allows_distinct_equal_valued_parameters(optimizer_cls):
    first = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    second = Tensor(np.array([1.0, 2.0]), requires_grad=True)

    optimizer = optimizer_cls([first, second])

    assert len(optimizer.parameters) == 2
    assert optimizer.parameters[0] is first
    assert optimizer.parameters[1] is second


def test_sgd_distinct_equal_parameters_each_update_once():
    first = Tensor(np.array([1.0]), requires_grad=True)
    second = Tensor(np.array([1.0]), requires_grad=True)
    first.grad = np.array([2.0])
    second.grad = np.array([2.0])

    optimizer = SGD([first, second], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(first.data, np.array([0.8]))
    np.testing.assert_allclose(second.data, np.array([0.8]))
