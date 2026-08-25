"""Structural regressions for tensorized beam sibling scoring."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


def _model(context_len=5):
    np.random.seed(61)
    return GPT(
        vocab_size=9,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        norm="layernorm",
        pos_encoding="learned",
        ffn="gelu",
    )


def test_uncached_selected_siblings_share_one_prefill_per_step():
    model = _model(context_len=5)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    shapes = []
    original_infer = model.infer

    def recording_infer(tokens, *args, **kwargs):
        shapes.append(np.asarray(tokens).shape)
        return original_infer(tokens, *args, **kwargs)

    model.infer = recording_infer
    beam_generate(model, prompt, 4, beam_width=2, use_cache=False)

    # The prompt is scored once. Each later selected sibling set is then
    # re-prefilled as one batch. Once sequence length exceeds context_len,
    # strict-window cropping keeps the per-row inference width at five.
    assert shapes == [(1, 3), (2, 4), (2, 5), (2, 5)]
