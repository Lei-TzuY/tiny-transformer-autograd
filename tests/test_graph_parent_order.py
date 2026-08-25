"""Computational-graph parent order must be stable and identity-deduplicated."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def test_children_preserve_first_seen_identity_order():
    first = Tensor(1.0, requires_grad=True)
    second = Tensor(2.0, requires_grad=True)
    node = Tensor(
        0.0,
        requires_grad=True,
        _children=(first, second, first),
        _op="probe",
    )

    assert isinstance(node._children, tuple)
    assert node._children == (first, second)


def test_backward_visits_sibling_graph_nodes_in_forward_parent_order():
    leaf = Tensor(2.0, requires_grad=True)
    left = leaf * 2.0
    right = leaf * 3.0
    root = left + right

    order = []
    left_backward = left._backward
    right_backward = right._backward

    def record_left():
        order.append("left")
        left_backward()

    def record_right():
        order.append("right")
        right_backward()

    left._backward = record_left
    right._backward = record_right
    root.backward()

    assert order == ["left", "right"]
    np.testing.assert_allclose(leaf.grad, 5.0)


def test_duplicate_operand_still_accumulates_each_derivative_contribution():
    x = Tensor([1.5, -2.0], requires_grad=True)
    y = x + x

    # The traversal stores one graph edge for one Tensor identity, while the
    # operation's VJP still contributes once for each operand occurrence.
    assert y._children == (x,)
    y.backward(np.array([2.0, -3.0]))
    np.testing.assert_allclose(x.grad, np.array([4.0, -6.0]))
