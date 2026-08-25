"""Autograd must reject tensor-data mutations that invalidate a saved graph."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import SGD
from engine.tensor import Tensor
from nn.module import Module


def _assert_clean_failure(output, leaf):
    before = leaf.grad.copy()
    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()
    np.testing.assert_array_equal(leaf.grad, before)


def test_leaf_setitem_after_forward_fails_before_gradient_mutation():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    x.data[0] = 10.0
    _assert_clean_failure(y, x)


def test_leaf_slice_view_mutation_is_tracked():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 4.0
    view = x.data[1:]
    view += 5.0
    _assert_clean_failure(y, x)


def test_leaf_ufunc_out_mutation_is_tracked():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    np.add(x.data, 1.0, out=x.data)
    _assert_clean_failure(y, x)


def test_replacing_data_after_forward_is_tracked():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    x.data = np.array([4.0, 5.0])
    _assert_clean_failure(y, x)


def test_intermediate_mutation_invalidates_downstream_graph():
    x = Tensor([2.0, 3.0], requires_grad=True)
    hidden = x * x
    output = hidden * 2.0
    hidden.data[0] = 100.0
    _assert_clean_failure(output, x)


def test_numpy_view_mutation_is_tracked():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    x.numpy()[1] = -4.0
    _assert_clean_failure(y, x)


def test_mutation_before_new_graph_is_allowed():
    x = Tensor([2.0, 3.0], requires_grad=True)
    x.data[0] = 5.0
    y = x * x
    y.backward(np.array([1.0, 2.0]))
    np.testing.assert_allclose(x.grad, np.array([10.0, 12.0]))


def test_repeated_backward_without_mutation_still_accumulates():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    y.backward()
    first = x.grad.copy()
    y.backward()
    np.testing.assert_allclose(x.grad, 2.0 * first)


def test_mutation_after_first_backward_rejects_reusing_old_graph_transactionally():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    y.backward()
    first = x.grad.copy()
    x.data += 1.0
    _assert_clean_failure(y, x)
    np.testing.assert_array_equal(x.grad, first)


def test_independent_copy_does_not_invalidate_graph():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    copied = x.data.copy()
    copied[:] = 99.0
    y.backward()
    np.testing.assert_allclose(x.grad, np.array([4.0, 6.0]))


def test_optimizer_step_invalidates_old_graph_but_new_forward_is_valid():
    x = Tensor([2.0, 3.0], requires_grad=True)
    old_graph = x * x
    old_graph.backward()

    optimizer = SGD([x], lr=0.1)
    optimizer.step()
    after_step = np.asarray(x.data).copy()
    _assert_clean_failure(old_graph, x)

    optimizer.zero_grad()
    new_graph = x * x
    new_graph.backward()
    np.testing.assert_allclose(x.grad, 2.0 * after_step)


class _OneParameter(Module):
    def __init__(self):
        self.weight = Tensor([2.0, 3.0], requires_grad=True)

    def forward(self):
        return self.weight * self.weight


def test_state_load_invalidates_old_graph_but_new_forward_is_valid():
    module = _OneParameter()
    old_graph = module()
    state = module.state_dict()
    state["weight"] = np.array([5.0, 7.0])
    module.load_state_dict(state)

    _assert_clean_failure(old_graph, module.weight)
    module.zero_grad()
    new_graph = module()
    new_graph.backward()
    np.testing.assert_allclose(module.weight.grad, np.array([10.0, 14.0]))
