"""JVP checker tolerances should normalize real-to-float overflow."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import jvpcheck
from engine.tensor import Tensor


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("name", ["eps", "atol", "rtol"])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_unrepresentable_integer_tolerance_is_finite_value_error_before_forward(
    name,
    value,
):
    x = Tensor([1.0, 2.0])
    tangent = np.array([0.0, 0.0])
    calls = []
    np.random.seed(321)
    before = np.random.get_state()

    def function(arg):
        calls.append(None)
        return arg

    with pytest.raises(ValueError, match=rf"jvpcheck {name} must be finite"):
        jvpcheck(function, x, tangent, **{name: value})

    assert calls == []
    _assert_rng_state_equal(np.random.get_state(), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": 1e300},
        {"atol": 1e300},
        {"rtol": 1e300},
    ],
)
def test_large_representable_tolerances_remain_supported(kwargs):
    x = Tensor([1.0, 2.0])
    tangent = np.zeros(2, dtype=np.float64)

    assert jvpcheck(lambda arg: arg, x, tangent, **kwargs)
