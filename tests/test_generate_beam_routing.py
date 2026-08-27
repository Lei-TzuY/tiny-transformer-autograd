"""Route ``GPT.generate(strategy='beam')`` through the public fused beam helper."""

import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.beam import beam_generate
from nn.transformer import GPT


def _model(*, pos_encoding="learned"):
    np.random.seed(2026)
    return GPT(
        vocab_size=13,
        context_len=5,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        pos_encoding=pos_encoding,
    )


def test_generate_beam_routes_batched_ragged_mask_to_public_helper():
    model = _model()
    idx = np.array([[0, 2, 4], [3, 5, 7]])
    mask = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)

    expected = beam_generate(
        model,
        idx,
        3,
        beam_width=2,
        temperature=0.8,
        attention_mask=mask,
        use_cache=True,
    )
    actual = model.generate(
        idx,
        3,
        strategy="beam",
        beam_width=2,
        temperature=0.8,
        attention_mask=mask,
        use_cache=True,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 6)


def test_generate_beam_forwards_uncached_execution_to_public_helper():
    model = _model(pos_encoding="rope")
    idx = np.array([[0, 1, 6], [2, 3, 8]])
    mask = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)

    expected = beam_generate(
        model,
        idx,
        3,
        beam_width=2,
        temperature=1.0,
        attention_mask=mask,
        use_cache=False,
    )
    actual = model.generate(
        idx,
        3,
        strategy="beam",
        beam_width=2,
        attention_mask=mask,
        use_cache=False,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("pos_encoding", ["learned", "rope"])
def test_generate_beam_keeps_single_prompt_reference_parity(pos_encoding):
    model = _model(pos_encoding=pos_encoding)
    idx = np.array([[1, 4, 6]])

    expected = model.generate_beam(idx, 2, beam_width=2, temperature=0.9)
    actual = model.generate(
        idx,
        2,
        strategy="beam",
        beam_width=2,
        temperature=0.9,
    )

    np.testing.assert_array_equal(actual, expected)


def test_generate_beam_forwards_validated_public_options_exactly():
    model = _model()
    idx = np.array([[0, 1, 2], [0, 3, 4]])
    mask = np.array([[0, 1, 1], [0, 1, 1]], dtype=bool)
    sentinel = np.array([[9]])

    with mock.patch("nn.beam.beam_generate", return_value=sentinel) as routed:
        result = model.generate(
            idx,
            4,
            strategy="beam",
            beam_width=3,
            temperature=np.float64(0.75),
            attention_mask=mask,
            use_cache=np.bool_(False),
        )

    assert result is sentinel
    routed.assert_called_once()
    args, kwargs = routed.call_args
    assert args[0] is model
    np.testing.assert_array_equal(args[1], idx)
    assert args[2] == 4
    assert kwargs["beam_width"] == 3
    assert kwargs["temperature"] == 0.75
    np.testing.assert_array_equal(kwargs["attention_mask"], mask)
    assert kwargs["use_cache"] is False


def test_generate_beam_zero_tokens_validates_mask_without_inference():
    model = _model()
    idx = np.array([[0, 2, 4], [3, 5, 7]])
    mask = np.array([[0, 1, 1], [1, 1, 1]], dtype=bool)

    with mock.patch.object(
        model,
        "infer",
        side_effect=AssertionError("zero-token beam generation must not infer"),
    ):
        result = model.generate(
            idx,
            0,
            strategy="beam",
            beam_width=2,
            attention_mask=mask,
        )

    np.testing.assert_array_equal(result, idx)
    assert result is not idx


def test_generate_beam_zero_tokens_still_rejects_malformed_mask():
    model = _model()
    idx = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="left-padded"):
        model.generate(
            idx,
            0,
            strategy="beam",
            attention_mask=np.array([[1, 1, 0]], dtype=np.int64),
        )
