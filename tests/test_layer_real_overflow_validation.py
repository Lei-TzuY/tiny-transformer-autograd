"""Layer real-valued hyperparameters should normalize float conversion overflow."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Dropout, LayerNorm, Linear, RMSNorm


HUGE = 10**400


def _assert_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize("layer_cls", [LayerNorm, RMSNorm])
@pytest.mark.parametrize("eps", [HUGE, -HUGE])
def test_norm_eps_conversion_overflow_is_a_finite_value_error(layer_cls, eps):
    with pytest.raises(ValueError, match=r"^eps must be finite$"):
        layer_cls(4, eps=eps)


@pytest.mark.parametrize("probability", [HUGE, -HUGE])
def test_dropout_probability_conversion_overflow_is_a_finite_value_error(probability):
    with pytest.raises(ValueError, match=r"^dropout probability must be finite$"):
        Dropout(probability)


@pytest.mark.parametrize("alpha", [HUGE, -HUGE])
def test_linear_lora_alpha_overflow_rejects_before_mutation_or_rng_use(alpha):
    np.random.seed(2026)
    layer = Linear(3, 2)
    weight = layer.weight
    bias = layer.bias
    weight_before = weight.data.copy()
    bias_before = bias.data.copy()
    rng_before = np.random.get_state()

    with pytest.raises(ValueError, match=r"^LoRA alpha must be finite$"):
        layer.enable_lora(2, alpha=alpha)

    assert layer.weight is weight
    assert layer.bias is bias
    np.testing.assert_array_equal(layer.weight.data, weight_before)
    np.testing.assert_array_equal(layer.bias.data, bias_before)
    assert layer.weight.requires_grad is True
    assert layer.bias.requires_grad is True
    assert layer.lora_A is None
    assert layer.lora_B is None
    assert layer.lora_scaling == 1.0
    _assert_rng_state_equal(np.random.get_state(), rng_before)


def test_large_representable_norm_eps_remains_supported():
    assert LayerNorm(2, eps=1e300).eps == 1e300
    assert RMSNorm(2, eps=1e300).eps == 1e300


def test_large_representable_dropout_uses_existing_range_validation():
    with pytest.raises(ValueError, match=r"^dropout probability must be less than 1.0$"):
        Dropout(1e300)


def test_large_representable_lora_alpha_remains_supported():
    np.random.seed(7)
    layer = Linear(2, 2)

    layer.enable_lora(2, alpha=1e300)

    assert layer.lora_A is not None
    assert layer.lora_B is not None
    assert layer.lora_scaling == 5e299
