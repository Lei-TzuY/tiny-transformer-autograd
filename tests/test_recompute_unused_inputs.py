"""Regression coverage for unused activation-recompute inputs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.recompute import recompute
from engine.tensor import Tensor


def test_recompute_skips_unused_requires_grad_input():
    x = Tensor([2.0, -3.0], requires_grad=True)
    unused = Tensor([5.0, 7.0], requires_grad=True)

    out = recompute(lambda used, ignored: (used * used).sum(), x, unused)
    out.backward()

    np.testing.assert_array_equal(x.grad, np.array([4.0, -6.0]))
    assert unused.grad is None


def test_recompute_preserves_existing_grad_on_unused_input():
    x = Tensor([1.0, 4.0], requires_grad=True)
    unused = Tensor([8.0, 9.0], requires_grad=True)
    existing = np.array([3.0, -2.0])
    unused.grad = existing.copy()

    out = recompute(lambda used, ignored: (used * 2.0).sum(), x, unused)
    out.backward()

    np.testing.assert_array_equal(x.grad, np.array([2.0, 2.0]))
    np.testing.assert_array_equal(unused.grad, existing)


def test_recompute_multi_output_skips_unused_input():
    x = Tensor([1.0, 2.0], requires_grad=True)
    unused = Tensor([10.0, 20.0], requires_grad=True)

    first, second = recompute(
        lambda used, ignored: (used * 2.0, used * 3.0),
        x,
        unused,
    )
    (first.sum() + second.sum()).backward()

    np.testing.assert_array_equal(x.grad, np.array([5.0, 5.0]))
    assert unused.grad is None


def test_recompute_unused_input_does_not_block_parameter_gradient():
    x = Tensor([2.0, 4.0], requires_grad=True)
    unused = Tensor([6.0, 8.0], requires_grad=True)
    weight = Tensor([3.0, 5.0], requires_grad=True)

    out = recompute(lambda used, ignored: (used * weight).sum(), x, unused)
    out.backward()

    np.testing.assert_array_equal(x.grad, weight.data)
    np.testing.assert_array_equal(weight.grad, x.data)
    assert unused.grad is None


def test_recompute_duplicate_input_only_accumulates_used_replay_copy():
    x = Tensor([2.0, -1.0], requires_grad=True)

    out = recompute(lambda used, ignored: (used * 4.0).sum(), x, x)
    out.backward()

    np.testing.assert_array_equal(x.grad, np.array([4.0, 4.0]))
