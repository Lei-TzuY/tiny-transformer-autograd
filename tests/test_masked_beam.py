"""Tests for left-padded strict-window beam search."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


ARCHITECTURES = [
    pytest.param(
        dict(norm="layernorm", pos_encoding="learned", ffn="gelu"),
        id="learned",
    ),
    pytest.param(
        dict(norm="rmsnorm", pos_encoding="rope", ffn="swiglu"),
        id="rope",
    ),
]


def _model(architecture, context_len=6):
    np.random.seed(23)
    return GPT(
        vocab_size=9,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        **architecture,
    )


def _left_pad(prompt, width, pad_token=0):
    tokens = np.full((1, width), pad_token, dtype=np.int64)
    mask = np.zeros((1, width), dtype=np.int64)
    tokens[0, -len(prompt):] = prompt
    mask[0, -len(prompt):] = 1
    return tokens, mask


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_unmasked_helper_matches_existing_beam_search(architecture):
    model = _model(architecture)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)

    expected = model.generate_beam(prompt, 4, beam_width=3, temperature=0.9)
    actual = beam_generate(model, prompt, 4, beam_width=3, temperature=0.9)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_left_padded_beam_matches_unpadded_prompt(architecture):
    model = _model(architecture)
    prompt = [1, 4, 2]
    tokens, mask = _left_pad(prompt, width=5)
    new_tokens = 4

    padded = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=3,
        attention_mask=mask,
    )
    plain = beam_generate(
        model,
        np.array([prompt], dtype=np.int64),
        new_tokens,
        beam_width=3,
    )

    np.testing.assert_array_equal(padded[0, -new_tokens:], plain[0, -new_tokens:])
    np.testing.assert_array_equal(padded[0, : tokens.shape[1]], tokens[0])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cached_and_uncached_masked_beam_match(architecture):
    model = _model(architecture)
    tokens, mask = _left_pad([1, 4, 2], width=5)

    cached = beam_generate(
        model,
        tokens,
        4,
        beam_width=3,
        attention_mask=mask,
        use_cache=True,
    )
    uncached = beam_generate(
        model,
        tokens,
        4,
        beam_width=3,
        attention_mask=mask,
        use_cache=False,
    )

    np.testing.assert_array_equal(cached, uncached)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_masked_beam_stays_equivalent_after_window_crop(architecture):
    model = _model(architecture, context_len=5)
    prompt = [1, 4, 2]
    tokens, mask = _left_pad(prompt, width=5)
    new_tokens = 7

    padded = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=2,
        attention_mask=mask,
    )
    plain = beam_generate(
        model,
        np.array([prompt], dtype=np.int64),
        new_tokens,
        beam_width=2,
    )

    assert padded.shape[1] > model.context_len
    np.testing.assert_array_equal(padded[0, -new_tokens:], plain[0, -new_tokens:])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cached_and_uncached_match_after_window_crop(architecture):
    model = _model(architecture, context_len=5)
    tokens, mask = _left_pad([1, 4, 2], width=5)
    new_tokens = 7

    cached = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=2,
        attention_mask=mask,
        use_cache=True,
    )
    uncached = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=2,
        attention_mask=mask,
        use_cache=False,
    )

    np.testing.assert_array_equal(cached, uncached)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_overlong_masked_prompt_crops_to_same_real_window(architecture):
    model = _model(architecture, context_len=5)
    prompt = [1, 4, 2, 6, 3, 7]
    tokens, mask = _left_pad(prompt, width=8)
    new_tokens = 3

    padded = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=2,
        attention_mask=mask,
    )
    plain = beam_generate(
        model,
        np.array([prompt], dtype=np.int64),
        new_tokens,
        beam_width=2,
    )

    np.testing.assert_array_equal(padded[0, -new_tokens:], plain[0, -new_tokens:])


def test_cached_beams_extend_one_token_until_strict_window_fills():
    model = _model(ARCHITECTURES[0].values[0], context_len=5)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    widths = []
    original_infer = model.infer

    def recording_infer(tokens, *args, **kwargs):
        widths.append(np.asarray(tokens).shape[1])
        return original_infer(tokens, *args, **kwargs)

    model.infer = recording_infer
    beam_generate(model, prompt, 4, beam_width=2, use_cache=True)

    # Initial prefill at width 3; each of the two selected beams then extends
    # one token twice. Once both caches reach context_len=5, strict semantics
    # require a full width-5 re-prefill for their next selected children.
    assert widths == [3, 1, 1, 1, 1, 5, 5]


def test_padding_token_values_cannot_change_beam_result():
    model = _model(ARCHITECTURES[0].values[0])
    tokens, mask = _left_pad([1, 4, 2], width=6)

    first = beam_generate(model, tokens, 4, beam_width=3, attention_mask=mask)
    scrambled = tokens.copy()
    scrambled[mask == 0] = 8
    second = beam_generate(model, scrambled, 4, beam_width=3, attention_mask=mask)

    np.testing.assert_array_equal(first[:, -4:], second[:, -4:])


def test_zero_tokens_returns_a_copy_after_mask_validation():
    model = _model(ARCHITECTURES[0].values[0])
    tokens, mask = _left_pad([1, 2], width=4)

    result = beam_generate(model, tokens, 0, attention_mask=mask)

    np.testing.assert_array_equal(result, tokens)
    assert result is not tokens


@pytest.mark.parametrize(
    ("tokens", "mask", "message"),
    [
        (
            np.array([[1, 2, 0]], dtype=np.int64),
            np.array([[1, 1, 0]], dtype=np.int64),
            "left-padded",
        ),
        (
            np.array([[0, 0]], dtype=np.int64),
            np.array([[0, 0]], dtype=np.int64),
            "at least one real token",
        ),
    ],
)
def test_rejects_invalid_generation_masks(tokens, mask, message):
    model = _model(ARCHITECTURES[0].values[0])

    with pytest.raises(ValueError, match=message):
        beam_generate(model, tokens, 2, attention_mask=mask)


def test_rejects_batched_beam_search_even_with_a_valid_mask():
    model = _model(ARCHITECTURES[0].values[0])
    tokens = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    mask = np.array([[0, 1, 1], [1, 1, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="batch size 1"):
        beam_generate(model, tokens, 2, attention_mask=mask)


def test_rejects_non_boolean_cache_switch():
    model = _model(ARCHITECTURES[0].values[0])

    with pytest.raises(TypeError, match="use_cache must be boolean"):
        beam_generate(model, np.array([[1, 2]]), 2, use_cache=1)
