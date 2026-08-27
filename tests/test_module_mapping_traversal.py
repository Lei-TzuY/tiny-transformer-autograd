"""Module traversal must descend through general Mapping containers."""

from collections import UserDict
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.module import Module


class _Leaf(Module):
    def __init__(self):
        self.training = True
        self.weight = Tensor([1.0, 2.0], requires_grad=True)
        self.buffer = Tensor([3.0], requires_grad=False)

    def forward(self, x):
        return x * self.weight


class _MappingParent(Module):
    def __init__(self):
        self.training = True
        self.children = UserDict({"leaf": _Leaf()})

    def forward(self, x):
        return self.children["leaf"](x)


def test_parameters_and_named_parameters_descend_into_userdict():
    module = _MappingParent()
    leaf = module.children["leaf"]

    assert module.parameters() == [leaf.weight]
    assert list(module.named_parameters()) == [("children[leaf].weight", leaf.weight)]
    assert module.param_count() == 2


def test_named_tensors_and_state_dict_include_mapping_children():
    module = _MappingParent()
    leaf = module.children["leaf"]

    named = dict(module.named_tensors())
    assert named == {
        "children[leaf].weight": leaf.weight,
        "children[leaf].buffer": leaf.buffer,
    }

    state = module.state_dict()
    np.testing.assert_array_equal(state["children[leaf].weight"], [1.0, 2.0])
    np.testing.assert_array_equal(state["children[leaf].buffer"], [3.0])


def test_load_state_dict_restores_mapping_child_tensors():
    module = _MappingParent()
    state = module.state_dict()
    state["children[leaf].weight"] = np.array([8.0, 9.0])
    state["children[leaf].buffer"] = np.array([10.0])

    module.load_state_dict(state)

    leaf = module.children["leaf"]
    np.testing.assert_array_equal(leaf.weight.data, [8.0, 9.0])
    np.testing.assert_array_equal(leaf.buffer.data, [10.0])


def test_eval_and_train_propagate_through_userdict():
    module = _MappingParent()
    leaf = module.children["leaf"]

    assert module.eval() is module
    assert module.training is False
    assert leaf.training is False

    assert module.train() is module
    assert module.training is True
    assert leaf.training is True


def test_zero_grad_reaches_parameter_inside_userdict():
    module = _MappingParent()
    leaf = module.children["leaf"]
    leaf.weight.grad[:] = [5.0, 7.0]

    module.zero_grad()

    np.testing.assert_array_equal(leaf.weight.grad, [0.0, 0.0])


def test_cyclic_userdict_is_traversed_once_without_recursion():
    module = _MappingParent()
    leaf = module.children["leaf"]
    module.children["cycle"] = module.children
    module.children["duplicate"] = leaf

    assert module.modules() == [module, leaf]
    assert module.parameters() == [leaf.weight]
    assert list(module.named_parameters()) == [("children[leaf].weight", leaf.weight)]
    assert list(module.named_tensors()) == [
        ("children[leaf].weight", leaf.weight),
        ("children[leaf].buffer", leaf.buffer),
    ]
