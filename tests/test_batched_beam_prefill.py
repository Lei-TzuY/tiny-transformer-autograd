"""Structural tests for the shared beam-search prompt prefill."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


def _model(context_len=6):
    np.random.seed(31)
    return GPT(
        vocab_size=10,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
    )


def _ragged_batch():
    tokens = np.array(
        [
            [0, 0, 1, 4, 2],
            [0, 3, 6, 1, 5],
            [7, 2, 8, 4, 1],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    return tokens, mask


def test_one_token_ragged_batch_uses_one_shared_prefill():
    model = _model()
    tokens, mask = _ragged_batch()
    calls = []
    original_infer = model.infer

    def recording_infer(current, *args, **kwargs):
        calls.append(tuple(np.asarray(current).shape))
        return original_infer(current, *args, **kwargs)

    model.infer = recording_infer
    result = beam_generate(
        model,
        tokens,
        max_new_tokens=1,
        beam_width=3,
        attention_mask=mask,
    )

    assert result.shape == (3, 6)
    assert calls == [(3, 5)]


def test_shared_prefill_preserves_independent_row_results():
    model = _model(context_len=5)
    tokens, mask = _ragged_batch()
    new_tokens = 6

    batched = beam_generate(
        model,
        tokens,
        new_tokens,
        beam_width=2,
        attention_mask=mask,
        use_cache=True,
    )

    prompts = [[1, 4, 2], [3, 6, 1, 5], [7, 2, 8, 4, 1]]
    for row, prompt in enumerate(prompts):
        alone = beam_generate(
            model,
            np.array([prompt], dtype=np.int64),
            new_tokens,
            beam_width=2,
            use_cache=True,
        )
        np.testing.assert_array_equal(
            batched[row, -new_tokens:],
            alone[0, -new_tokens:],
        )
