"""Counterexample for treating a shifted RoPE KV cache as an exact re-prefill.

RoPE makes a global position shift cheap: cached keys can be rotated back by
one position after the oldest key is dropped.  That is sufficient for the
first Transformer block because its K/V projections depend only on the token
embedding entering that block.

It is not sufficient for deeper blocks.  Their cached K/V tensors were built
from hidden states that had already attended to the now-removed oldest token.
A strict sliding-window re-prefill recomputes those hidden states without that
token, while a ring/shift cache preserves the old influence.  The two policies
therefore have different semantics once the model has more than one block.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.transformer import GPT


def _model(num_layers):
    np.random.seed(21)
    model = GPT(
        vocab_size=16,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=num_layers,
        pos_encoding="rope",
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

    # This is the key counterexample: position rebasing is correct, yet stale
    # higher-layer K/V still carry information from the dropped token.
    assert difference > 1e-8
