import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.module import Module


class _Leaf(Module):
    def __init__(self, value):
        self.weight = Tensor(np.array([value, value + 1.0]), requires_grad=True)
        self.buffer = Tensor(np.array([value + 2.0]), requires_grad=False)


class _Root(Module):
    def __init__(self):
        self.weight = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        self.buffer = Tensor(np.array([4.0]), requires_grad=False)


def test_self_reference_does_not_recurse_through_public_traversals():
    root = _Root()
    root.self_ref = root

    assert root.modules() == [root]
    assert root.parameters() == [root.weight]
    assert list(root.named_parameters()) == [("weight", root.weight)]
    assert list(root.named_tensors()) == [
        ("weight", root.weight),
        ("buffer", root.buffer),
    ]
    assert root.param_count() == 3

    state = root.state_dict()
    assert list(state) == ["weight", "buffer"]

    root.weight.data[:] = -1.0
    root.buffer.data[:] = -2.0
    root.load_state_dict(state)
    np.testing.assert_array_equal(root.weight.data, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(root.buffer.data, [4.0])


def test_mutual_module_cycle_keeps_first_path_as_canonical_name():
    root = _Root()
    child = _Leaf(10.0)
    root.first = child
    root.second = child
    child.parent = root

    assert root.modules() == [root, child]
    assert root.parameters() == [root.weight, child.weight]
    assert list(root.named_parameters()) == [
        ("weight", root.weight),
        ("first.weight", child.weight),
    ]
    assert list(root.named_tensors()) == [
        ("weight", root.weight),
        ("buffer", root.buffer),
        ("first.weight", child.weight),
        ("first.buffer", child.buffer),
    ]
    assert root.param_count() == root.weight.data.size + child.weight.data.size

    state = root.state_dict()
    assert set(state) == {"weight", "buffer", "first.weight", "first.buffer"}
    assert not any("parent" in name or "second" in name for name in state)


def test_container_cycles_and_shared_containers_are_traversed_once():
    root = _Root()
    child = _Leaf(20.0)

    registry = [child]
    mapping = {"child": child, "registry": registry}
    registry.extend([registry, mapping])
    mapping["self"] = mapping

    root.registry = registry
    root.registry_alias = registry

    assert root.modules() == [root, child]
    assert root.parameters() == [root.weight, child.weight]
    assert list(root.named_parameters()) == [
        ("weight", root.weight),
        ("registry[0].weight", child.weight),
    ]
    assert list(root.named_tensors()) == [
        ("weight", root.weight),
        ("buffer", root.buffer),
        ("registry[0].weight", child.weight),
        ("registry[0].buffer", child.buffer),
    ]


def test_train_and_eval_terminate_on_cyclic_module_graphs():
    root = _Root()
    child = _Leaf(30.0)
    root.child = child
    child.parent = root

    assert root.train(False) is root
    assert root.training is False
    assert child.training is False

    assert root.train(True) is root
    assert root.training is True
    assert child.training is True

    assert root.eval() is root
    assert root.training is False
    assert child.training is False


def test_zero_grad_touches_each_shared_parameter_once_semantically():
    root = _Root()
    child = _Leaf(40.0)
    root.left = child
    root.right = child
    child.parent = root

    root.weight.grad[:] = 7.0
    child.weight.grad[:] = 9.0
    root.zero_grad()

    np.testing.assert_array_equal(root.weight.grad, np.zeros_like(root.weight.grad))
    np.testing.assert_array_equal(child.weight.grad, np.zeros_like(child.weight.grad))


def test_repr_terminates_on_direct_self_reference():
    root = _Root()
    root.self_ref = root

    rendered = repr(root)

    assert rendered.startswith("_Root(\n")
    assert "  (self_ref): ..." in rendered
    assert rendered.endswith("\n)")


def test_repr_terminates_on_mutual_module_cycle():
    root = _Root()
    child = _Leaf(50.0)
    root.child = child
    child.parent = root

    rendered = repr(root)

    assert "  (child): _Leaf(" in rendered
    assert "  (parent): ..." in rendered


def test_repr_for_acyclic_modules_keeps_existing_format():
    root = _Root()
    child = _Leaf(60.0)
    root.child = child

    rendered = repr(root)

    assert rendered.startswith("_Root(\n")
    assert "  (child): _Leaf(" in rendered
    assert "..." not in rendered
    assert rendered.endswith("\n)")
