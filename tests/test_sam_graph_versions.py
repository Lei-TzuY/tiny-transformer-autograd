import numpy as np
import pytest

from engine.ops import sum as tensor_sum
from engine.optim import SGD
from engine.sam import SAM
from engine.tensor import Tensor


def test_first_step_invalidates_graph_built_at_base_weights():
    parameter = Tensor([2.0], requires_grad=True)
    old_loss = tensor_sum(parameter * parameter)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.1)

    optimizer.first_step()

    with pytest.raises(RuntimeError, match="modified after forward"):
        old_loss.backward()
    optimizer.restore()


def test_restore_does_not_resurrect_graph_built_before_perturbation():
    parameter = Tensor([2.0], requires_grad=True)
    old_loss = tensor_sum(parameter * parameter)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter]), rho=0.1)

    optimizer.first_step()
    optimizer.restore()
    np.testing.assert_array_equal(parameter.data, [2.0])

    with pytest.raises(RuntimeError, match="modified after forward"):
        old_loss.backward()


def test_second_step_invalidates_graph_built_at_perturbed_weights_after_backward():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter], lr=0.1), rho=0.1)
    optimizer.first_step()
    optimizer.zero_grad()

    neighbourhood_loss = tensor_sum(parameter * parameter)
    neighbourhood_loss.backward()
    optimizer.second_step()

    with pytest.raises(RuntimeError, match="modified after forward"):
        neighbourhood_loss.backward()


def test_zero_radius_first_step_keeps_existing_graph_valid_until_second_step():
    parameter = Tensor([2.0], requires_grad=True)
    loss = tensor_sum(parameter * parameter)
    parameter.grad[...] = [1.0]
    optimizer = SAM(SGD([parameter], lr=0.1), rho=0.0)
    version = parameter._version

    optimizer.first_step()
    assert parameter._version == version

    # No parameter write occurred, so the original graph is still valid.
    parameter.grad[...] = [0.0]
    loss.backward()
    optimizer.second_step()

    assert parameter._version > version
