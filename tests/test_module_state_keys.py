"""State-dictionary key validation tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Linear


def test_state_dict_rejects_non_string_keys_before_key_set_sorting():
    layer = Linear(2, 2)
    state = layer.state_dict()
    state[1] = state["weight"].copy()

    with pytest.raises(TypeError, match="keys must be strings"):
        layer.load_state_dict(state, strict=False)
