"""Ownership and immutability regressions for cached beam search."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.beam import _extend_cache, _prefill, _slice_cache
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
    np.random.seed(53)
    return GPT(
        vocab_size=9,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        **architecture,
    )


def _snapshot(cache):
    return [
        (entry["k"].copy(), entry["v"].copy())
        for entry in cache
    ]


def _assert_unchanged(cache, snapshot):
    for entry, (key, value) in zip(cache, snapshot):
        np.testing.assert_array_equal(entry["k"], key)
        np.testing.assert_array_equal(entry["v"], value)


def _assert_read_only(cache):
    for entry in cache:
        assert not entry["k"].flags.writeable
        assert not entry["v"].flags.writeable


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_gpt_infer_does_not_mutate_parent_cache_or_alias_child(architecture):
    model = _model(architecture)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    _, parent = model.infer(prompt)
    before = _snapshot(parent)

    _, child = model.infer(np.array([[6]], dtype=np.int64), parent)

    _assert_unchanged(parent, before)
    for parent_entry, child_entry in zip(parent, child):
        assert child_entry["k"].shape[2] == parent_entry["k"].shape[2] + 1
        assert child_entry["v"].shape[2] == parent_entry["v"].shape[2] + 1
        assert not np.shares_memory(parent_entry["k"], child_entry["k"])
        assert not np.shares_memory(parent_entry["v"], child_entry["v"])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_batched_prefill_rows_are_zero_copy_read_only_views(architecture):
    model = _model(architecture)
    prompts = np.array([[1, 4, 2], [3, 6, 7]], dtype=np.int64)

    _, cache = _prefill(model, prompts, None)
    row = _slice_cache(cache, 1)

    _assert_read_only(cache)
    _assert_read_only(row)
    for full_entry, row_entry in zip(cache, row):
        assert np.shares_memory(full_entry["k"], row_entry["k"])
        assert np.shares_memory(full_entry["v"], row_entry["v"])

    with pytest.raises(ValueError, match="read-only"):
        row[0]["k"][...] = 0.0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_child_extension_preserves_parent_and_returns_frozen_storage(architecture):
    model = _model(architecture)
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    _, parent = _prefill(model, prompt, None)
    before = _snapshot(parent)
    sequence = np.array([[1, 4, 2, 6]], dtype=np.int64)

    _, child = _extend_cache(model, sequence, parent, None)

    _assert_read_only(parent)
    _assert_read_only(child)
    _assert_unchanged(parent, before)
    for parent_entry, child_entry in zip(parent, child):
        assert not np.shares_memory(parent_entry["k"], child_entry["k"])
        assert not np.shares_memory(parent_entry["v"], child_entry["v"])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_beam_tree_never_passes_writable_parent_cache_to_infer(architecture):
    model = _model(architecture)
    original_infer = model.infer
    cached_calls = 0

    def guarded_infer(
        tokens,
        kv_cache=None,
        attention_mask=None,
        position_ids=None,
    ):
        nonlocal cached_calls
        before = None
        if kv_cache is not None:
            cached_calls += 1
            _assert_read_only(kv_cache)
            before = _snapshot(kv_cache)

        result = original_infer(
            tokens,
            kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if kv_cache is not None:
            _assert_unchanged(kv_cache, before)
        return result

    model.infer = guarded_infer
    prompt = np.array([[1, 4, 2]], dtype=np.int64)
    beam_generate(model, prompt, 4, beam_width=2, use_cache=True)

    assert cached_calls > 0
