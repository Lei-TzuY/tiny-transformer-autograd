"""Ownership regressions for streaming RoPE cache rebasing."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.streaming import _drop_oldest_and_rebase
from nn.transformer import GPT


def _model():
    np.random.seed(67)
    return GPT(
        vocab_size=11,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        norm="rmsnorm",
        pos_encoding="rope",
        ffn="swiglu",
    )


def test_drop_and_rebase_never_mutates_or_aliass_parent_cache():
    model = _model()
    prompt = np.array(
        [
            [1, 3, 5, 7],
            [0, 0, 2, 4],
        ],
        dtype=np.int64,
    )
    keep = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    positions = np.array(
        [
            [0, 1, 2, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    _, parent = model.infer(
        prompt,
        attention_mask=keep,
        position_ids=positions,
    )
    before = [
        (entry["k"].copy(), entry["v"].copy())
        for entry in parent
    ]

    # Row 0 drops a real token and must rebase surviving RoPE keys. Row 1
    # drops padding, so its surviving keys must keep their existing rotation.
    shifted = _drop_oldest_and_rebase(
        parent,
        model,
        np.array([True, False]),
    )

    for entry, (key_before, value_before), new_entry in zip(parent, before, shifted):
        np.testing.assert_array_equal(entry["k"], key_before)
        np.testing.assert_array_equal(entry["v"], value_before)
        assert not np.shares_memory(entry["k"], new_entry["k"])
        assert not np.shares_memory(entry["v"], new_entry["v"])
        np.testing.assert_array_equal(new_entry["v"], value_before[:, :, 1:, :])
        np.testing.assert_array_equal(
            new_entry["k"][1],
            key_before[1, :, 1:, :],
        )

    # The returned streaming state is independently owned: later changes to it
    # cannot retroactively corrupt the full parent cache used to derive it.
    shifted[0]["k"][...] = 0.0
    shifted[0]["v"][...] = 0.0
    np.testing.assert_array_equal(parent[0]["k"], before[0][0])
    np.testing.assert_array_equal(parent[0]["v"], before[0][1])
