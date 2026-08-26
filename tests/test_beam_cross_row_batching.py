"""Structural regressions for cross-row beam inference fusion."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


def _model(context_len=6):
    np.random.seed(71)
    return GPT(
        vocab_size=11,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        norm="rmsnorm",
        pos_encoding="rope",
        ffn="swiglu",
    )


def _ragged_prompts():
    tokens = np.array(
        [
            [0, 1, 4, 2],
            [0, 0, 3, 6],
            [0, 0, 0, 7],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    return tokens, mask


@pytest.mark.parametrize(
    ("use_cache", "expected_shapes"),
    [
        (True, [(3, 4), (6, 1), (6, 1)]),
        (False, [(3, 4), (6, 5), (6, 6)]),
    ],
)
def test_selected_beams_from_all_prompt_rows_share_each_infer(
    use_cache,
    expected_shapes,
):
    model = _model(context_len=6)
    tokens, mask = _ragged_prompts()
    shapes = []
    original_infer = model.infer

    def recording_infer(batch, *args, **kwargs):
        shapes.append(np.asarray(batch).shape)
        return original_infer(batch, *args, **kwargs)

    model.infer = recording_infer
    beam_generate(
        model,
        tokens,
        3,
        beam_width=2,
        attention_mask=mask,
        use_cache=use_cache,
    )

    # Candidate ranking remains independent per prompt row, but after each row
    # selects two children all 3*2 children are scored as one inference batch.
    assert shapes == expected_shapes
