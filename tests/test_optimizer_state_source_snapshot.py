"""Regression tests for aliased optimizer state sources during restore."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, AdamW, SGD
from engine.tensor import Tensor


class _ChangingBuffers(list):
    """List-shaped state source that changes values on a second iteration."""

    def __init__(self, first, second):
        super().__init__(first)
        self.first = first
        self.second = second
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        values = self.first if self.iterations == 1 else self.second
        return iter(values)


def _parameter():
    return Tensor([1.0, -2.0], requires_grad=True)


def test_sgd_load_snapshots_aliased_velocity_sources_before_commit():
    optimizer = SGD([_parameter(), _parameter()], momentum=0.9)
    optimizer._v[0][:] = [1.0, 2.0]
    optimizer._v[1][:] = [10.0, 20.0]
    first_buffer = optimizer._v[0]
    second_buffer = optimizer._v[1]

    state = optimizer.state_dict()
    state["v"] = [optimizer._v[1], optimizer._v[0]]
    optimizer.load_state_dict(state)

    assert optimizer._v[0] is first_buffer
    assert optimizer._v[1] is second_buffer
    np.testing.assert_array_equal(optimizer._v[0], [10.0, 20.0])
    np.testing.assert_array_equal(optimizer._v[1], [1.0, 2.0])


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_load_snapshots_cross_group_moment_aliases_before_commit(optimizer_cls):
    optimizer = optimizer_cls([_parameter()])
    optimizer._m[0][:] = [1.0, 2.0]
    optimizer._v[0][:] = [10.0, 20.0]
    first_moment = optimizer._m[0]
    second_moment = optimizer._v[0]

    state = optimizer.state_dict()
    state["m"] = [optimizer._v[0]]
    state["v"] = [optimizer._m[0]]
    optimizer.load_state_dict(state)

    assert optimizer._m[0] is first_moment
    assert optimizer._v[0] is second_moment
    np.testing.assert_array_equal(optimizer._m[0], [10.0, 20.0])
    np.testing.assert_array_equal(optimizer._v[0], [1.0, 2.0])


def test_buffer_container_is_materialized_once_before_validation_and_copy():
    optimizer = SGD([_parameter()], momentum=0.9)
    good = np.array([3.0, 4.0])
    bad = np.array([np.nan, 99.0])
    changing = _ChangingBuffers([good], [bad])
    state = optimizer.state_dict()
    state["v"] = changing

    optimizer.load_state_dict(state)

    assert changing.iterations == 1
    np.testing.assert_array_equal(optimizer._v[0], good)


def test_adam_late_invalid_group_still_cannot_partially_commit_snapshots():
    optimizer = Adam([_parameter()])
    optimizer._m[0][:] = [1.0, 2.0]
    optimizer._v[0][:] = [10.0, 20.0]
    before = optimizer.state_dict()

    state = optimizer.state_dict()
    state["m"] = [optimizer._v[0]]
    state["v"][0][0] = np.nan

    with pytest.raises(ValueError, match="second moment.*finite"):
        optimizer.load_state_dict(state)

    after = optimizer.state_dict()
    np.testing.assert_array_equal(after["m"][0], before["m"][0])
    np.testing.assert_array_equal(after["v"][0], before["v"][0])
