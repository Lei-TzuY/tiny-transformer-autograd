"""Streaming-cache parity, cost, and semantic-boundary tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.streaming import stream_generate
from nn.transformer import GPT


def _model(num_layers, pos_encoding="rope"):
    np.random.seed(21)
    model = GPT(
        vocab_size=16,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=num_layers,
        pos_encoding=pos_encoding,
    )
    # The normal tiny initialization makes the semantic difference very small.
    # Scaling the learned matrices keeps the same architecture while making the
    # counterexample comfortably larger than floating-point noise.
    for parameter in model.parameters():
        parameter.data *= 5.0
    return model


def _rotate_half_np(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _drop_oldest_and_rebase_rope(cache, model):
    """Drop one cache slot and rotate surviving keys from p to p-1."""
    cos = model.rope.cos[1]
    sin = model.rope.sin[1]
    shifted = []
    for entry in cache:
        key = entry["k"][:, :, 1:, :]
        value = entry["v"][:, :, 1:, :]
        # R(-1)x = cos(1)x - sin(1)Jx, where J is RoPE's rotate-half.
        key = key * cos - _rotate_half_np(key) * sin
        shifted.append({"k": key, "v": value})
    return shifted


def _next_logits(model):
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    new_token = np.array([[2]], dtype=np.int64)

    _, full_cache = model.infer(prompt)
    shifted_cache = _drop_oldest_and_rebase_rope(full_cache, model)
    streamed, _ = model.infer(
        new_token,
        shifted_cache,
        position_ids=np.array([[model.context_len - 1]], dtype=np.int64),
    )

    strict_window = np.concatenate([prompt[:, 1:], new_token], axis=1)
    exact, _ = model.infer(strict_window)
    return exact[:, -1, :], streamed[:, -1, :]


def test_shifted_rope_cache_is_exact_for_one_block():
    exact, streamed = _next_logits(_model(num_layers=1))
    np.testing.assert_allclose(streamed, exact, atol=1e-12, rtol=1e-12)


def test_shifted_rope_cache_is_not_an_exact_multiblock_replacement():
    exact, streamed = _next_logits(_model(num_layers=2))
    difference = float(np.max(np.abs(streamed - exact)))

    # Position rebasing is correct, yet stale higher-layer K/V still carry
    # information from the dropped token.
    assert difference > 1e-8


def test_one_block_streaming_generation_matches_strict_window():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    exact = _model(1).generate(prompt, 8, strategy="greedy")
    streamed = stream_generate(_model(1), prompt, 8, strategy="greedy")

    np.testing.assert_array_equal(streamed, exact)


def test_one_block_masked_streaming_matches_strict_window():
    prompt = np.array(
        [
            [0, 1, 3, 5],
            [0, 0, 2, 4],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )

    exact = _model(1).generate(
        prompt,
        7,
        strategy="greedy",
        attention_mask=mask,
    )
    streamed = stream_generate(
        _model(1),
        prompt,
        7,
        strategy="greedy",
        attention_mask=mask,
    )

    np.testing.assert_array_equal(streamed, exact)


def test_streaming_stays_incremental_after_the_window_fills():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    model = _model(2)
    widths = []
    infer = model.infer

    def counted(idx, *args, **kwargs):
        widths.append(np.asarray(idx).shape[1])
        return infer(idx, *args, **kwargs)

    model.infer = counted
    stream_generate(model, prompt, 6, strategy="greedy")

    # One prompt prefill, then one-token cache extensions only. The strict
    # generate() path would re-prefill four positions at every post-window step.
    assert widths == [4, 1, 1, 1, 1, 1]


def test_zero_new_tokens_returns_a_copy_without_inference():
    prompt = np.array([[1, 3]], dtype=np.int64)
    model = _model(1)

    def fail_infer(*_args, **_kwargs):
        raise AssertionError("zero-token generation must not run inference")

    model.infer = fail_infer
    result = stream_generate(model, prompt, 0, strategy="greedy")

    np.testing.assert_array_equal(result, prompt)
    assert result is not prompt


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: stream_generate(
                _model(1, pos_encoding="learned"),
                np.array([[1, 2]]),
                1,
            ),
            ValueError,
            "pos_encoding='rope'",
        ),
        (
            lambda: stream_generate(_model(1), np.array([[1, 2]]), -1),
            ValueError,
            "non-negative integer",
        ),
        (
            lambda: stream_generate(
                _model(1), np.array([[1, 2]]), 1, strategy="beam"
            ),
            ValueError,
            "sample.*greedy",
        ),
        (
            lambda: stream_generate(
                _model(1),
                np.array([[0, 1]]),
                1,
                attention_mask=np.array([[1, 0]]),
            ),
            ValueError,
            "left-padded",
        ),
    ],
)
def test_streaming_validates_its_contract(call, error, message):
    with pytest.raises(error, match=message):
        call()
