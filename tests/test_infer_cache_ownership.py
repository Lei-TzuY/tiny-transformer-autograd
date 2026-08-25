"""Core KV-cache ownership regressions for ``GPT.infer``."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.transformer import GPT


def _model():
    np.random.seed(2026)
    return GPT(
        vocab_size=11,
        context_len=6,
        d_model=4,
        num_heads=2,
        d_ff=8,
        num_layers=2,
        dropout=0.0,
    )


def _snapshot(cache):
    return [
        {"k": entry["k"].copy(), "v": entry["v"].copy()}
        for entry in cache
    ]


def test_infer_branches_from_read_only_parent_without_aliasing():
    model = _model()
    _, parent = model.infer(np.array([[1, 2, 3]], dtype=np.int64))
    before = _snapshot(parent)

    for entry in parent:
        entry["k"].flags.writeable = False
        entry["v"].flags.writeable = False

    _, child_a = model.infer(np.array([[4]], dtype=np.int64), parent)
    _, child_b = model.infer(np.array([[5]], dtype=np.int64), parent)

    assert len(parent) == len(child_a) == len(child_b) == model.num_layers
    for layer, (parent_entry, a_entry, b_entry, saved) in enumerate(
        zip(parent, child_a, child_b, before)
    ):
        np.testing.assert_array_equal(parent_entry["k"], saved["k"])
        np.testing.assert_array_equal(parent_entry["v"], saved["v"])
        assert not parent_entry["k"].flags.writeable
        assert not parent_entry["v"].flags.writeable

        assert a_entry["k"].shape[2] == parent_entry["k"].shape[2] + 1
        assert a_entry["v"].shape[2] == parent_entry["v"].shape[2] + 1
        np.testing.assert_array_equal(
            a_entry["k"][:, :, :-1], parent_entry["k"]
        )
        np.testing.assert_array_equal(
            a_entry["v"][:, :, :-1], parent_entry["v"]
        )

        for name in ("k", "v"):
            assert not np.shares_memory(a_entry[name], parent_entry[name]), layer
            assert not np.shares_memory(b_entry[name], parent_entry[name]), layer
            assert not np.shares_memory(a_entry[name], b_entry[name]), layer

    child_b_before = _snapshot(child_b)
    child_a[0]["k"][...] = 123.0
    child_a[0]["v"][...] = -456.0

    for parent_entry, saved in zip(parent, before):
        np.testing.assert_array_equal(parent_entry["k"], saved["k"])
        np.testing.assert_array_equal(parent_entry["v"], saved["v"])
    for child_entry, saved in zip(child_b, child_b_before):
        np.testing.assert_array_equal(child_entry["k"], saved["k"])
        np.testing.assert_array_equal(child_entry["v"], saved["v"])
