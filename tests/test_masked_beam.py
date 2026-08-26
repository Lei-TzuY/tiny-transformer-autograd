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


def _left_padded_batch(prompts, width=None, pad_token=0):
    width = width or max(len(prompt) for prompt in prompts)
    tokens = np.full((len(prompts), width), pad_token, dtype=np.int64)
    mask = np.zeros((len(prompts), width), dtype=np.int64)
    for row, prompt in enumerate(prompts):
        tokens[row, -len(prompt):] = prompt
        mask[row, -len(prompt):] = 1
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
def test_ragged_batched_beam_matches_each_prompt_alone(architecture):
    model = _model(architecture)
    prompts = [[1, 4, 2], [3, 6], [7]]
    tokens, mask = _left_padded_batch(prompts, width=4)
    new_tokens = 4

    batched = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=3,
        attention_mask=mask,
    )

    assert batched.shape == (len(prompts), tokens.shape[1] + new_tokens)
    np.testing.assert_array_equal(batched[:, : tokens.shape[1]], tokens)
    for row, prompt in enumerate(prompts):
        alone = beam_generate(
            model,
            np.array([prompt], dtype=np.int64),
            new_tokens,
            beam_width=3,
        )
        np.testing.assert_array_equal(
            batched[row, -new_tokens:],
            alone[0, -new_tokens:],
            err_msg=f"row {row} beam result differs from solo decoding",
        )


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
def test_ragged_batch_cache_parity_after_window_crop(architecture):
    model = _model(architecture, context_len=5)
    prompts = [[1, 4, 2], [3, 6], [7]]
    tokens, mask = _left_padded_batch(prompts, width=5)
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

    assert cached.shape[1] > model.context_len
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


def test_cached_beam_siblings_share_one_batched_infer_per_step():
    model = _model(ARCHITECTURES[0].values[0], context_len=5)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    shapes = []
    original_infer = model.infer

    def recording_infer(tokens, *args, **kwargs):
        shapes.append(np.asarray(tokens).shape)
        return original_infer(tokens, *args, **kwargs)

    model.infer = recording_infer
    beam_generate(model, prompt, 4, beam_width=2, use_cache=True)

    # One prompt prefill. The two selected siblings are then scored together
    # while cache extension fits, and are re-prefilled together once the strict
    # context window is full. The old path made one infer call per sibling.
    assert shapes == [(1, 3), (2, 1), (2, 1), (2, 5)]


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


def test_rejects_non_boolean_cache_switch():
    model = _model(ARCHITECTURES[0].values[0])

    with pytest.raises(TypeError, match="use_cache must be boolean"):
        beam_generate(model, np.array([[1, 2]]), 2, use_cache=1)
