"""Module-level examples for parameter-aware finite-difference gradcheck."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.gradcheck import gradcheck
from engine.tensor import Tensor
from nn.layers import LayerNorm, Linear, RMSNorm


def test_gradcheck_linear_lora_parameters_only():
    np.random.seed(31)
    layer = Linear(2, 2)
    layer.enable_lora(rank=1, alpha=2.0)
    x = Tensor([[0.4, -0.7], [1.2, 0.3]])

    names = [name for name, _ in layer.named_parameters()]
    assert names == ["lora_A", "lora_B"]
    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=2e-6,
        rtol=2e-5,
    )


def test_gradcheck_layernorm_affine_parameters():
    layer = LayerNorm(3)
    x = Tensor([[0.2, -0.4, 1.1], [0.8, 0.3, -0.6]])

    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=3e-6,
        rtol=3e-5,
    )


def test_gradcheck_rmsnorm_scale_parameter():
    layer = RMSNorm(3)
    x = Tensor([[0.2, -0.4, 1.1], [0.8, 0.3, -0.6]])

    assert gradcheck(
        lambda value: layer(value),
        x,
        parameters=layer.named_parameters(),
        atol=3e-6,
        rtol=3e-5,
    )
