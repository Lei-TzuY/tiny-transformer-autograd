"""Current-main temperature-scaling parity for direct beam generation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


def _model():
    np.random.seed(83)
    return GPT(
        vocab_size=3,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
    )


def test_direct_beam_reuses_stable_tiny_temperature_scaling():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    def extreme_infer(idx, *_args, **_kwargs):
        batch, time = np.asarray(idx).shape
        logits = np.empty((batch, time, model.vocab_size), dtype=np.float64)
        logits[..., 0] = 1e308
        logits[..., 1] = 0.0
        logits[..., 2] = -1e308
        return logits, []

    model.infer = extreme_infer
    temperature = np.float64(1e-308)

    expected = model.generate_beam(
        prompt,
        1,
        beam_width=2,
        temperature=temperature,
    )
    actual = beam_generate(
        model,
        prompt,
        1,
        beam_width=2,
        temperature=temperature,
        use_cache=False,
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, -1] == 0
