"""Regression coverage for Tensor data mutations through ``ndarray.flat``."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor


def _assert_clean_failure(output, leaf):
    before = leaf.grad.copy()
    with pytest.raises(RuntimeError, match="modified after forward"):
        output.backward()
    np.testing.assert_array_equal(leaf.grad, before)


def test_flat_index_mutation_invalidates_existing_graph_transactionally():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    output = x * x

    x.data.flat[1] = 20.0

    np.testing.assert_array_equal(x.data, np.array([1.0, 20.0, 3.0]))
    _assert_clean_failure(output, x)


def test_flat_slice_mutation_invalidates_existing_graph():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    output = x * 3.0

    x.data.flat[1:3] = [8.0, 9.0]

    np.testing.assert_array_equal(x.data, np.array([1.0, 8.0, 9.0, 4.0]))
    _assert_clean_failure(output, x)


def test_flat_attribute_assignment_invalidates_existing_graph():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    output = x * x

    x.data.flat = [7.0, 8.0]

    np.testing.assert_array_equal(x.data, np.array([7.0, 8.0, 7.0, 8.0]))
    _assert_clean_failure(output, x)


def test_flat_mutation_through_shared_view_invalidates_owner_graph():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    output = x * x
    view = x.data[1:]

    view.flat[1] = -5.0

    np.testing.assert_array_equal(x.data, np.array([1.0, 2.0, -5.0, 4.0]))
    _assert_clean_failure(output, x)


def test_flat_reads_preserve_iterator_behavior_without_invalidating_graph():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    output = x * 2.0
    flat = x.data.flat

    assert len(flat) == 4
    assert flat[2] == 3.0
    np.testing.assert_array_equal(flat[1:3], np.array([2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(flat), np.array([1.0, 2.0, 3.0, 4.0]))
    assert next(flat) == 1.0
    assert flat.index == 1
    assert flat.coords == (0, 1)

    output.backward()
    np.testing.assert_array_equal(x.grad, np.full((2, 2), 2.0))


def test_independent_copy_flat_mutation_does_not_invalidate_graph():
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    output = x * x
    copied = x.data.copy()

    copied.flat[0] = 99.0

    np.testing.assert_array_equal(x.data, np.array([2.0, 3.0, 4.0]))
    output.backward()
    np.testing.assert_array_equal(x.grad, np.array([4.0, 6.0, 8.0]))
