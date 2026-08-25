"""Persistent tensor names must be unambiguous for state serialization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.module import Module


def _ambiguous_module():
    module = Module()
    first = Tensor([1.0])
    second = Tensor([2.0])
    module.slots = {1: first, "1": second}
    return module, first, second


def test_named_tensors_rejects_distinct_tensors_with_same_serialized_name():
    module, _, _ = _ambiguous_module()

    with pytest.raises(ValueError, match="ambiguous persistent tensor name"):
        list(module.named_tensors())


def test_state_dict_rejects_ambiguous_tensor_names_instead_of_overwriting():
    module, _, _ = _ambiguous_module()

    with pytest.raises(ValueError, match=r"slots\[1\]"):
        module.state_dict()


def test_load_state_dict_rejects_ambiguous_destinations_before_mutation():
    module, first, second = _ambiguous_module()
    first_before = first.data.copy()
    second_before = second.data.copy()

    with pytest.raises(ValueError, match=r"slots\[1\]"):
        module.load_state_dict({"slots[1]": np.array([9.0])}, strict=False)

    np.testing.assert_array_equal(first.data, first_before)
    np.testing.assert_array_equal(second.data, second_before)


def test_shared_tensor_alias_with_same_rendered_name_remains_deduplicated():
    module = Module()
    shared = Tensor([3.0])
    module.slots = {1: shared, "1": shared}

    named = list(module.named_tensors())
    state = module.state_dict()

    assert named == [("slots[1]", shared)]
    assert list(state) == ["slots[1]"]
    np.testing.assert_array_equal(state["slots[1]"], shared.data)
