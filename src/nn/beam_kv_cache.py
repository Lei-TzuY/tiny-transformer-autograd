"""Branchable beam search for :class:`GPTKVCache`.

The legacy :mod:`nn.beam` implementation owns immutable dictionary snapshots and
batches sibling caches by concatenating them along the batch axis.  A mutable
preallocated cache needs a different ownership rule: before a selected child can
append its token, it receives an independent fork of its parent's live K/V prefix.

This module intentionally starts with eager forks rather than copy-on-write storage.
That makes aliasing and failure semantics explicit: a fork never shares writable K/V
storage with its source, and mutating one beam cannot affect siblings or the parent.
"""

import numpy as np

from .gpt_kv_cache import GPTKVCache, _model_versions, infer_gpt_with_kv_cache
from .transformer import (
    GPT,
    _left_padded_positions,
    _log_softmax,
    _temperature_scale_logits,
    _validate_non_negative_int,
    _validate_positive_finite_real,
    _validate_positive_int,
)


def fork_gpt_kv_cache(cache):
    """Return an independently writable logical fork of ``cache``.

    A non-empty source is required to still match the exact model tensor-version
    snapshot that produced its K/V values.  The live prefix is copied into fresh
    fixed-capacity layer buffers; no K/V storage is shared with the source.  An empty
    source forks to a fresh empty cache because it has no logical K/V state to copy.
    """
    if not isinstance(cache, GPTKVCache):
        raise TypeError("cache must be a GPTKVCache")

    with cache._lock:
        initialized, length = cache._state_unlocked()
        if length > 0:
            if cache._model_versions is None:
                raise RuntimeError("non-empty GPT cache is missing model-version metadata")
            current_versions = _model_versions(cache._model)
            if current_versions != cache._model_versions:
                raise RuntimeError("model tensors changed while GPT KV cache was live")

        snapshots = None
        if initialized and length > 0:
            snapshots = [buffer.snapshot() for buffer in cache._buffers]
        versions = cache._model_versions
        model = cache._model

    child = GPTKVCache(model)
    if snapshots is None:
        return child

    # Build the entire fork privately.  If an allocation/write fails, the source is
    # untouched and the partially constructed child is unreachable from the caller.
    for buffer, entry in zip(child._buffers, snapshots):
        buffer.append(entry["k"], entry["v"])
    child._model_versions = tuple(versions)
    initialized, child_length = child._state_unlocked()
    if not initialized or child_length != length:
        raise RuntimeError("forked GPT KV cache postcondition failed")
    return child


def beam_generate_gpt_with_kv_cache(
    model,
    idx,
    max_new_tokens,
    *,
    beam_width=3,
    temperature=1.0,
    attention_mask=None,
    cache=None,
):
    """Generate one best beam using independently forked preallocated KV caches.

    The current implementation deliberately matches ``GPT.generate_beam``'s batch-size
    one contract.  It returns ``(tokens, best_cache)``; unlike many beam helpers, the
    returned cache already includes the final generated token and can therefore be
    passed directly to :func:`infer_gpt_with_kv_cache` for continuation.

    ``attention_mask`` follows normal generation semantics: a single left-padded row
    with 0/False padding and 1/True real tokens.  When the strict context window fills,
    each selected child clears its private allocation and refills from its cropped
    window so forgotten tokens do not survive in attention state.
    """
    if not isinstance(model, GPT):
        raise TypeError("model must be a GPT")
    max_new_tokens = _validate_non_negative_int(max_new_tokens, "max_new_tokens")
    beam_width = _validate_positive_int(beam_width, "beam_width")
    temperature = _validate_positive_finite_real(temperature, "temperature")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    if idx.shape[0] != 1:
        raise ValueError("buffered beam search currently supports batch size 1")

    if cache is None:
        cache = GPTKVCache(model)
    elif not isinstance(cache, GPTKVCache):
        raise TypeError("cache must be a GPTKVCache or None")
    cache._assert_model(model)
    if cache.length != 0:
        raise ValueError("beam generation cache must be empty at entry")

    mask = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape).copy()

    logits = _prefill(model, idx, mask, cache)
    beams = [(idx, 0.0, logits, cache, mask)]

    for _ in range(max_new_tokens):
        candidates = []
        for sequence, score, beam_logits, parent_cache, beam_mask in beams:
            scaled = _temperature_scale_logits(beam_logits[0, -1], temperature)
            log_probs = _log_softmax(scaled)
            best = np.argsort(log_probs)[-beam_width:]
            for token in best:
                extended = np.concatenate([sequence, [[token]]], axis=1)
                extended_mask = None
                if beam_mask is not None:
                    extended_mask = np.concatenate(
                        [beam_mask, np.ones((1, 1), dtype=bool)],
                        axis=1,
                    )
                candidates.append(
                    (
                        extended,
                        score + float(log_probs[token]),
                        parent_cache,
                        extended_mask,
                    )
                )

        selected = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )[:beam_width]
        if not selected:
            raise RuntimeError("beam search produced no candidates")

        advanced = []
        for sequence, score, parent_cache, beam_mask in selected:
            child = fork_gpt_kv_cache(parent_cache)
            child_logits = _advance_child(model, sequence, beam_mask, child)
            advanced.append((sequence, score, child_logits, child, beam_mask))
        beams = advanced

    best_sequence, _, _, best_cache, _ = beams[0]
    return best_sequence, best_cache


def _prefill(model, sequence, mask, cache):
    window = sequence[:, -model.context_len :]
    window_mask = positions = None
    if mask is not None:
        width = window.shape[1]
        window_mask = mask[:, -width:]
        positions = _left_padded_positions(window_mask)
    logits, _ = infer_gpt_with_kv_cache(
        model,
        window,
        cache,
        attention_mask=window_mask,
        position_ids=positions,
    )
    return logits


def _advance_child(model, sequence, mask, cache):
    if cache.length < model.context_len:
        step_mask = step_positions = None
        if mask is not None:
            cached = cache.length
            step_mask = mask[:, -(cached + 1) :]
            step_positions = _left_padded_positions(step_mask)[:, -1:]
        logits, _ = infer_gpt_with_kv_cache(
            model,
            sequence[:, -1:],
            cache,
            attention_mask=step_mask,
            position_ids=step_positions,
        )
        return logits

    # Strict-window semantics: the selected token belongs to a new cropped window.
    # Clearing retains the child's already allocated layer storage for refill.
    cache.clear()
    return _prefill(model, sequence, mask, cache)
