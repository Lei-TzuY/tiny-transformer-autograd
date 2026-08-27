"""Public-boundary validation for optimizer parameter collections."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


OPTIMIZERS = [
    pytest.param(lambda parameters: SGD(parameters), id="sgd"),
    pytest.param(lambda parameters: Adam(parameters), id="adam"),
    pytest.param(lambda parameters: AdamW(parameters), id="adamw"),
]


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_optimizer_rejects_non_iterable_parameters_explicitly(factory):
    with pytest.raises(TypeError, match="parameters must be an iterable of Tensors"):
        factory(object())


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_optimizer_rejects_non_tensor_parameter_with_index(factory):
    valid = Tensor([1.0], requires_grad=True)

    with pytest.raises(TypeError, match="optimizer parameter 1 must be a Tensor"):
        factory([valid, object()])


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_parameter_type_validation_precedes_data_access(factory):
    class DataPoison:
        @property
        def data(self):
            pytest.fail("invalid optimizer parameter data must not be inspected")

    with pytest.raises(TypeError, match="optimizer parameter 0 must be a Tensor"):
        factory([DataPoison()])


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_generator_parameters_are_materialized_once_and_preserve_order(factory):
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    yielded = []

    def parameters():
        yielded.append(first)
        yield first
        yielded.append(second)
        yield second

    optimizer = factory(parameters())

    assert optimizer.parameters == [first, second]
    assert yielded == [first, second]


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_duplicate_tensor_rejection_remains_unchanged(factory):
    parameter = Tensor([1.0], requires_grad=True)

    with pytest.raises(ValueError, match="duplicate at index 1"):
        factory([parameter, parameter])
