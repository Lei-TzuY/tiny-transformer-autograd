"""Regression tests for stable, alias-safe Module state loading."""

from collections.abc import Mapping
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
from nn.layers import Linear
from nn.module import Module


class _Pair(Module):
    def __init__(self):
        self.left = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        self.right = Tensor(np.array([7.0, 8.0]), requires_grad=True)


class _ChangingMapping(Mapping):
    """Expose a different value on a second items() observation."""

    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.items_calls = 0

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(("weight",))

    def __getitem__(self, key):
        if key != "weight":
            raise KeyError(key)
        return self.first

    def items(self):
        self.items_calls += 1
        value = self.first if self.items_calls == 1 else self.second
        return [("weight", value)]


def test_live_destination_sources_are_snapshotted_before_any_write():
    module = _Pair()
    left_before = module.left.data.copy()
    right_before = module.right.data.copy()

    # Both payloads are live views of destinations that will be overwritten.
    # Without source snapshots, writing left first destroys right's source and
    # both tensors end up with right_before instead of swapping values.
    module.load_state_dict(
        {
            "left": module.right.data,
            "right": module.left.data,
        }
    )

    np.testing.assert_array_equal(module.left.data, right_before)
    np.testing.assert_array_equal(module.right.data, left_before)


def test_dynamic_mapping_is_materialized_once_before_validation_and_commit():
    layer = Linear(2, 2, bias=False)
    first = np.full_like(layer.weight.data, 0.25)
    second = np.full_like(layer.weight.data, np.nan)
    state = _ChangingMapping(first, second)

    layer.load_state_dict(state)

    assert state.items_calls == 1
    np.testing.assert_array_equal(layer.weight.data, first)
    assert np.isfinite(layer.weight.data).all()
