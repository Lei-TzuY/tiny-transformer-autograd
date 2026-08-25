"""Validation and compatibility tests for Module.load_state_dict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.layers import Linear
from nn.transformer import GPT


def test_valid_state_roundtrip_restores_all_tensors():
    np.random.seed(5)
    source = Linear(3, 2)
    state = source.state_dict()

    np.random.seed(9)
    target = Linear(3, 2)
    target.load_state_dict(state)

    for name, tensor in target.named_tensors():
        np.testing.assert_array_equal(tensor.data, state[name])


def test_late_invalid_value_leaves_every_tensor_unchanged():
    layer = Linear(3, 2)
    before = layer.state_dict()
    state = {
        "weight": np.full_like(layer.weight.data, 7.0),
        "bias": np.array([0.0, np.nan]),
    }

    with pytest.raises(ValueError, match="bias.*finite values"):
        layer.load_state_dict(state)

    for name, tensor in layer.named_tensors():
        np.testing.assert_array_equal(tensor.data, before[name])


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], TypeError, "NumPy array"),
        (np.array([[1, 2], [3, 4]], dtype=object), TypeError, "real numeric dtype"),
        (np.ones((2, 2), dtype=np.complex128), TypeError, "real numeric dtype"),
        (np.array([[1.0, np.inf], [3.0, 4.0]]), ValueError, "finite values"),
    ],
)
def test_rejects_invalid_parameter_payloads(value, error, message):
    layer = Linear(2, 2, bias=False)

    with pytest.raises(error, match=message):
        layer.load_state_dict({"weight": value})


def test_nonstrict_load_ignores_unknown_payload_without_inspecting_it():
    layer = Linear(2, 2)
    replacement = np.full_like(layer.weight.data, 0.25)

    layer.load_state_dict(
        {"weight": replacement, "future_metadata": object()},
        strict=False,
    )

    np.testing.assert_array_equal(layer.weight.data, replacement)


def test_legacy_finite_causal_mask_is_still_migrated_to_negative_infinity():
    model = GPT(
        vocab_size=7,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    )
    state = model.state_dict()
    state["causal_mask"] = np.triu(
        np.full((4, 4), -1e9, dtype=np.float64),
        k=1,
    )

    model.load_state_dict(state)

    upper = np.triu_indices(4, k=1)
    assert np.isneginf(model.causal_mask.data[upper]).all()
    np.testing.assert_array_equal(
        np.tril(model.causal_mask.data),
        np.zeros((4, 4), dtype=np.float64),
    )


def test_load_state_dict_validates_state_and_strict_argument_types():
    layer = Linear(2, 2)

    with pytest.raises(TypeError, match="must be a mapping"):
        layer.load_state_dict([("weight", layer.weight.data.copy())])
    with pytest.raises(TypeError, match="strict flag must be boolean"):
        layer.load_state_dict(layer.state_dict(), strict=1)
