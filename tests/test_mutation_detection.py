"""Autograd must reject tensor-data mutations that invalidate a saved graph."""

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
