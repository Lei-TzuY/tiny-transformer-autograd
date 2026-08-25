"""Regression tests for indexing state captured by Tensor.__getitem__."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def test_integer_index_array_mutation_cannot_reroute_backward():
    x = Tensor(np.arange(5.0), requires_grad=True)
    index = np.array([0, 2, 2], dtype=np.int64)

    selected = x[index]
    np.testing.assert_array_equal(selected.data, np.array([0.0, 2.0, 2.0]))

    # Mutating caller-owned indexing state after forward must not change the
    # VJP. Repeated index 2 should still receive both cotangent contributions.
    index[:] = [1, 1, 4]
    selected.backward(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(x.grad, np.array([1.0, 0.0, 5.0, 0.0, 0.0]))


def test_boolean_mask_mutation_cannot_reroute_backward():
    x = Tensor(np.array([10.0, 20.0, 30.0, 40.0]), requires_grad=True)
    mask = np.array([True, False, True, False])

    selected = x[mask]
    mask[:] = ~mask
    selected.backward(np.array([4.0, 7.0]))

    np.testing.assert_array_equal(x.grad, np.array([4.0, 0.0, 7.0, 0.0]))


def test_nested_tuple_and_list_indices_are_snapshotted_recursively():
    x = Tensor(np.arange(12.0).reshape(3, 4), requires_grad=True)
    rows = np.array([0, 2], dtype=np.int64)
    cols = [1, 3]
    index = (rows, cols)

    selected = x[index]
    rows[:] = [1, 1]
    cols[:] = [0, 0]
    selected.backward(np.array([2.0, 5.0]))

    expected = np.zeros((3, 4))
    expected[0, 1] = 2.0
    expected[2, 3] = 5.0
    np.testing.assert_array_equal(x.grad, expected)


def test_basic_slice_backward_semantics_are_unchanged():
    x = Tensor(np.arange(6.0), requires_grad=True)
    selected = x[1:5:2]
    selected.backward(np.array([3.0, 8.0]))

    np.testing.assert_array_equal(x.grad, np.array([0.0, 3.0, 0.0, 8.0, 0.0, 0.0]))
