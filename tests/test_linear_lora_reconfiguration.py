"""Regression coverage for repeated Linear LoRA configuration."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Linear


def _copy_rng_state():
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _assert_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _snapshot_layer(layer):
    return {
        "weight": layer.weight,
        "bias": layer.bias,
        "lora_A": layer.lora_A,
        "lora_B": layer.lora_B,
        "weight_data": layer.weight.data.copy(),
        "bias_data": None if layer.bias is None else layer.bias.data.copy(),
        "lora_A_data": layer.lora_A.data.copy(),
        "lora_B_data": layer.lora_B.data.copy(),
        "scaling": layer.lora_scaling,
    }


def _assert_layer_matches_snapshot(layer, snapshot):
    assert layer.weight is snapshot["weight"]
    assert layer.bias is snapshot["bias"]
    assert layer.lora_A is snapshot["lora_A"]
    assert layer.lora_B is snapshot["lora_B"]
    np.testing.assert_array_equal(layer.weight.data, snapshot["weight_data"])
    if layer.bias is not None:
        np.testing.assert_array_equal(layer.bias.data, snapshot["bias_data"])
    np.testing.assert_array_equal(layer.lora_A.data, snapshot["lora_A_data"])
    np.testing.assert_array_equal(layer.lora_B.data, snapshot["lora_B_data"])
    assert layer.lora_scaling == snapshot["scaling"]
    assert not layer.weight.requires_grad
    assert layer.weight.grad is None
    if layer.bias is not None:
        assert not layer.bias.requires_grad
        assert layer.bias.grad is None
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_repeating_same_lora_configuration_is_an_rng_free_noop():
    np.random.seed(41)
    layer = Linear(3, 4)
    layer.enable_lora(2, alpha=3.0)
    snapshot = _snapshot_layer(layer)
    rng_before = _copy_rng_state()

    layer.enable_lora(np.int64(2), alpha=np.float32(3.0))

    _assert_layer_matches_snapshot(layer, snapshot)
    _assert_rng_state_equal(np.random.get_state(), rng_before)


@pytest.mark.parametrize(
    "rank,alpha",
    [
        (3, 4.0),
        (2, 6.0),
    ],
)
def test_conflicting_lora_reconfiguration_is_rejected_transactionally(rank, alpha):
    np.random.seed(42)
    layer = Linear(3, 4)
    layer.enable_lora(2, alpha=4.0)
    snapshot = _snapshot_layer(layer)
    rng_before = _copy_rng_state()

    with pytest.raises(ValueError, match="LoRA adapters are already enabled"):
        layer.enable_lora(rank, alpha=alpha)

    _assert_layer_matches_snapshot(layer, snapshot)
    _assert_rng_state_equal(np.random.get_state(), rng_before)


def test_invalid_repeat_still_reports_input_validation_before_enabled_state():
    layer = Linear(3, 4)
    layer.enable_lora(2, alpha=4.0)

    with pytest.raises(ValueError, match="LoRA rank"):
        layer.enable_lora(0, alpha=4.0)
    with pytest.raises(TypeError, match="LoRA alpha"):
        layer.enable_lora(2, alpha="4.0")
