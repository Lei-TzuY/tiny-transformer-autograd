import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.module_summary import module_summary
from engine.tensor import Tensor
from nn.module import Module


def _rng_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class _Child(Module):
    def __init__(self):
        self.weight = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        self.buffer = Tensor([10.0, 20.0], requires_grad=False)


class _Toy(Module):
    def __init__(self):
        self.scalar = Tensor(3.0, requires_grad=True)
        self.empty = Tensor(np.empty((0, 4)), requires_grad=False)
        self.child = _Child()


def test_summary_counts_persistent_trainable_frozen_and_gradient_state_exactly():
    model = _Toy()

    report = module_summary(model)

    assert report["module_type"] == "_Toy"
    assert report["module_count"] == 2
    assert report["persistent_tensor_count"] == 4
    assert report["trainable_tensor_count"] == 2
    assert report["frozen_tensor_count"] == 2
    assert report["gradient_tensor_count"] == 2
    assert report["persistent_element_count"] == 9
    assert report["trainable_element_count"] == 7
    assert report["frozen_element_count"] == 2
    assert report["gradient_element_count"] == 7
    assert report["persistent_byte_count"] == 9 * 8
    assert report["trainable_byte_count"] == 7 * 8
    assert report["frozen_byte_count"] == 2 * 8
    assert report["gradient_byte_count"] == 7 * 8
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_tensor_entries_preserve_named_tensor_order_and_metadata():
    model = _Toy()
    model.child.weight.data[0, 0] = 99.0

    report = module_summary(model)
    entries = report["tensors"]

    assert [entry["name"] for entry in entries] == [
        "scalar",
        "empty",
        "child.weight",
        "child.buffer",
    ]
    assert entries[0] == {
        "name": "scalar",
        "shape": [],
        "dtype": "float64",
        "element_count": 1,
        "byte_count": 8,
        "requires_grad": True,
        "has_grad": True,
        "gradient_element_count": 1,
        "gradient_byte_count": 8,
        "mutation_version": 0,
    }
    assert entries[1]["shape"] == [0, 4]
    assert entries[1]["element_count"] == 0
    assert entries[1]["byte_count"] == 0
    assert entries[1]["requires_grad"] is False
    assert entries[1]["has_grad"] is False
    assert entries[1]["gradient_element_count"] == 0
    assert entries[1]["gradient_byte_count"] == 0
    assert entries[2]["mutation_version"] == 1


def test_gradient_totals_follow_actual_allocated_buffers_not_trainability():
    model = _Toy()
    model.scalar.grad = None
    model.child.buffer.grad = np.zeros_like(model.child.buffer.data)

    report = module_summary(model)

    assert report["trainable_tensor_count"] == 2
    assert report["gradient_tensor_count"] == 2
    assert report["gradient_element_count"] == 8
    assert report["gradient_byte_count"] == 8 * 8
    by_name = {entry["name"]: entry for entry in report["tensors"]}
    assert by_name["scalar"]["has_grad"] is False
    assert by_name["scalar"]["gradient_byte_count"] == 0
    assert by_name["child.buffer"]["requires_grad"] is False
    assert by_name["child.buffer"]["has_grad"] is True
    assert by_name["child.buffer"]["gradient_element_count"] == 2
    assert by_name["child.buffer"]["gradient_byte_count"] == 16


def test_shared_tensor_is_counted_once_through_module_traversal_contract():
    model = Module()
    shared = Tensor([1.0, 2.0], requires_grad=True)
    model.left = shared
    model.alias = shared

    report = module_summary(model)

    assert report["persistent_tensor_count"] == 1
    assert report["trainable_tensor_count"] == 1
    assert report["gradient_tensor_count"] == 1
    assert report["persistent_element_count"] == 2
    assert report["gradient_element_count"] == 2
    assert [entry["name"] for entry in report["tensors"]] == ["left"]


def test_summary_is_observational_for_tensor_grad_version_and_rng_state():
    model = _Toy()
    model.scalar.grad[...] = 7.0
    model.child.weight.grad[:] = np.arange(6.0).reshape(2, 3)

    tensors = tuple(model.named_tensors())
    data_before = [tensor.data.copy() for _, tensor in tensors]
    grad_objects = [tensor.grad for _, tensor in tensors]
    grad_before = [
        None if tensor.grad is None else tensor.grad.copy() for _, tensor in tensors
    ]
    versions_before = [tensor._version for _, tensor in tensors]

    np.random.seed(20260829)
    rng_before = np.random.get_state()

    module_summary(model)

    rng_after = np.random.get_state()
    assert _rng_state_equal(rng_before, rng_after)
    for index, (_, tensor) in enumerate(tensors):
        assert np.array_equal(tensor.data, data_before[index])
        assert tensor.grad is grad_objects[index]
        if grad_before[index] is not None:
            assert np.array_equal(tensor.grad, grad_before[index])
        assert tensor._version == versions_before[index]


def test_empty_module_has_zero_totals_and_valid_json():
    report = module_summary(Module())

    assert report["module_count"] == 1
    assert report["persistent_tensor_count"] == 0
    assert report["trainable_tensor_count"] == 0
    assert report["frozen_tensor_count"] == 0
    assert report["gradient_tensor_count"] == 0
    assert report["persistent_element_count"] == 0
    assert report["gradient_element_count"] == 0
    assert report["persistent_byte_count"] == 0
    assert report["gradient_byte_count"] == 0
    assert report["tensors"] == []
    json.dumps(report, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("bad", [None, object(), Tensor([1.0]), 123, "module"])
def test_public_validation_rejects_non_modules_explicitly(bad):
    with pytest.raises(TypeError, match="module_summary module must be an nn.Module"):
        module_summary(bad)
