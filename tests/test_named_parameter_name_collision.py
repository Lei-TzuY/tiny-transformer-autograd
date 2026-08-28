"""Named trainable parameters must never silently share one serialized path."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.module import Module


class _MappingParameters(Module):
    def __init__(self, values):
        self.params = values


def test_distinct_trainable_tensors_with_same_rendered_mapping_key_are_rejected():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    module = _MappingParameters({1: first, "1": second})

    # Identity-based parameter discovery is still complete. Only the ambiguous
    # public name is invalid: both mapping keys render as ``params[1]``.
    assert module.parameters() == [first, second]

    with pytest.raises(
        ValueError,
        match=r"^ambiguous trainable parameter name 'params\[1\]' maps to multiple tensors$",
    ):
        list(module.named_parameters())


def test_collision_is_detected_with_a_named_parameter_prefix():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    module = _MappingParameters({1: first, "1": second})

    with pytest.raises(
        ValueError,
        match=(
            r"^ambiguous trainable parameter name 'root\.params\[1\]' "
            r"maps to multiple tensors$"
        ),
    ):
        list(module.named_parameters(prefix="root"))


def test_same_tensor_under_colliding_rendered_keys_remains_identity_deduplicated():
    shared = Tensor([3.0, 4.0], requires_grad=True)
    module = _MappingParameters({1: shared, "1": shared})

    assert module.parameters() == [shared]
    assert list(module.named_parameters()) == [("params[1]", shared)]


def test_non_colliding_mapping_parameter_names_remain_deterministic():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    module = _MappingParameters({1: first, 2: second})

    assert list(module.named_parameters()) == [
        ("params[1]", first),
        ("params[2]", second),
    ]


def test_frozen_tensor_does_not_create_a_trainable_parameter_name_collision():
    trainable = Tensor([1.0], requires_grad=True)
    frozen = Tensor([2.0], requires_grad=False)
    module = _MappingParameters({1: trainable, "1": frozen})

    assert list(module.named_parameters()) == [("params[1]", trainable)]
    np.testing.assert_array_equal(trainable.data, [1.0])
    np.testing.assert_array_equal(frozen.data, [2.0])


def test_persistent_tensor_namespace_keeps_its_existing_collision_error():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([2.0], requires_grad=True)
    module = _MappingParameters({1: first, "1": second})

    with pytest.raises(
        ValueError,
        match=r"^ambiguous persistent tensor name 'params\[1\]' maps to multiple tensors$",
    ):
        list(module.named_tensors())
