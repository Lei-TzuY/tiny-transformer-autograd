"""Regression tests for layer constructor and LoRA hyperparameters."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Dropout, Embedding, LayerNorm, Linear, RMSNorm


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: Linear(value, 2),
        lambda value: Linear(2, value),
        lambda value: Embedding(value, 2),
        lambda value: Embedding(2, value),
        lambda value: LayerNorm(value),
        lambda value: RMSNorm(value),
    ],
)
@pytest.mark.parametrize(
    "bad_dimension",
    [0, -1, True, np.bool_(False), 1.5, np.float64(2.0), "2"],
)
def test_layer_dimensions_require_positive_nonboolean_integers(
    factory, bad_dimension
):
    with pytest.raises((TypeError, ValueError), match="positive|integer"):
        factory(bad_dimension)


def test_numpy_integer_dimensions_are_accepted_and_normalized():
    linear = Linear(np.int64(2), np.int32(3))
    embedding = Embedding(np.int32(5), np.int64(4))
    layer_norm = LayerNorm(np.int64(3))
    rms_norm = RMSNorm(np.int32(3))

    assert type(linear.in_features) is int
    assert type(linear.out_features) is int
    assert linear.weight.shape == (3, 2)
    assert type(embedding.num_embeddings) is int
    assert type(embedding.embedding_dim) is int
    assert embedding.weight.shape == (5, 4)
    assert type(layer_norm.normalized_shape) is int
    assert type(rms_norm.normalized_shape) is int


@pytest.mark.parametrize("norm_cls", [LayerNorm, RMSNorm])
@pytest.mark.parametrize(
    "bad_eps",
    [0.0, -1.0, np.nan, np.inf, -np.inf, True, np.bool_(False), "1e-5"],
)
def test_norms_reject_nonpositive_nonfinite_or_nonreal_eps(norm_cls, bad_eps):
    with pytest.raises((TypeError, ValueError), match="eps"):
        norm_cls(4, eps=bad_eps)


@pytest.mark.parametrize("norm_cls", [LayerNorm, RMSNorm])
def test_norms_accept_numpy_real_eps_and_keep_outputs_finite(norm_cls):
    norm = norm_cls(np.int64(3), eps=np.float32(1e-4))
    values = np.array([[1.0, -2.0, 0.5]])

    assert type(norm.eps) is float
    assert np.isfinite(norm.infer(values)).all()


@pytest.mark.parametrize(
    "bad_probability",
    [np.nan, np.inf, -np.inf, -0.1, 1.0, True, np.bool_(False), "0.2"],
)
def test_dropout_requires_a_finite_real_probability(bad_probability):
    with pytest.raises((TypeError, ValueError), match="dropout probability"):
        Dropout(bad_probability)


def test_dropout_accepts_numpy_real_probability_and_normalizes_it():
    dropout = Dropout(np.float32(0.25))
    assert type(dropout.p) is float
    assert dropout.p == pytest.approx(0.25)


@pytest.mark.parametrize(
    "bad_rank",
    [0, -1, True, np.bool_(False), 1.5, np.float64(2.0), "2"],
)
def test_lora_rank_is_validated_before_freezing_base_parameters(bad_rank):
    layer = Linear(3, 4)

    with pytest.raises((TypeError, ValueError), match="LoRA rank"):
        layer.enable_lora(bad_rank)

    assert layer.lora_A is None
    assert layer.lora_B is None
    assert layer.weight.requires_grad
    assert layer.weight.grad is not None
    assert layer.bias.requires_grad
    assert layer.bias.grad is not None


@pytest.mark.parametrize(
    "bad_alpha",
    [np.nan, np.inf, -np.inf, True, np.bool_(False), "1.0"],
)
def test_lora_alpha_is_validated_before_freezing_base_parameters(bad_alpha):
    layer = Linear(3, 4)

    with pytest.raises((TypeError, ValueError), match="LoRA alpha"):
        layer.enable_lora(2, alpha=bad_alpha)

    assert layer.lora_A is None
    assert layer.lora_B is None
    assert layer.weight.requires_grad
    assert layer.weight.grad is not None
    assert layer.bias.requires_grad
    assert layer.bias.grad is not None


def test_lora_accepts_numpy_scalars_and_normalizes_scaling():
    layer = Linear(3, 4)
    layer.enable_lora(np.int64(2), alpha=np.float32(3.0))

    assert layer.lora_A.shape == (2, 3)
    assert layer.lora_B.shape == (4, 2)
    assert layer.lora_scaling == 1.5
    assert not layer.weight.requires_grad
    assert layer.weight.grad is None
    assert not layer.bias.requires_grad
    assert layer.bias.grad is None
