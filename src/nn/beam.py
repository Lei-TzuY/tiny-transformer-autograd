"""Beam search helpers with explicit left-padding support.

``GPT.generate_beam`` predates ragged prompt masks and intentionally remains a
small unmasked reference path. ``beam_generate`` adds masked and batched forms
without changing that method, and can keep one KV cache per selected beam until
strict sliding-window semantics require a re-prefill.
"""

import numpy as np

from .transformer import _left_padded_positions, _log_softmax


def beam_generate(
    model,
    idx,
    max_new_tokens,
    beam_width=3,
    temperature=1.0,
    attention_mask=None,
    use_cache=True,
):
    """Return the best strict-window beam independently for every prompt row.

    ``attention_mask`` follows generation semantics: rows are left padded, 0
    marks padding, and 1 marks real prompt tokens. Generated tokens are always
    real. The returned array preserves the original shared padded width and
    appends exactly ``max_new_tokens`` columns.

    The prompt prefill is batched across every row, then each row receives its
    own logits and cache slice and runs an independent beam tree. Scores are
    never compared across prompts. With ``use_cache=True`` each selected beam
    in a tree owns the KV cache for its current strict window. Branches may
    safely share a parent cache because inference concatenates new arrays
    instead of mutating it. Once a cache fills ``context_len``, the next
    selected child is re-prefilled from its cropped window so positions are
    renumbered from zero and out-of-window tokens are genuinely forgotten.
    """
    if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not isinstance(beam_width, (int, np.integer)) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(use_cache, (bool, np.bool_)):
        raise TypeError("use_cache must be boolean")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    mask = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape).copy()
    if max_new_tokens == 0:
        return idx

    # Every row shares the same padded slot width, so the expensive prompt
    # prefill can be one batched inference even when real prompt lengths differ.
    # Per-row position_ids preserve ragged left-padding semantics.
    logits, cache = _prefill(model, idx, mask)

    rows = []
    for row in range(idx.shape[0]):
        row_mask = None if mask is None else mask[row : row + 1]
        row_cache = _slice_cache(cache, row) if use_cache else None
        rows.append(
            _beam_generate_one(
                model,
                idx[row : row + 1],
                max_new_tokens,
                beam_width,
                temperature,
                row_mask,
                use_cache,
                logits[row : row + 1],
                row_cache,
            )
        )
    return np.concatenate(rows, axis=0)


def _beam_generate_one(
    model,
    idx,
    max_new_tokens,
    beam_width,
    temperature,
    mask,
    use_cache,
    initial_logits,
    initial_cache,
):
    """Decode one validated prompt row from its batched-prefill state."""
    beams = [(idx, 0.0, initial_logits, initial_cache)]

    for step in range(max_new_tokens):
        candidates = []
        for sequence, score, logits, cache in beams:
            log_probs = _log_softmax(logits[0, -1] / temperature)
            best = np.argsort(log_probs)[-beam_width:]
            for token in best:
                extended = np.concatenate([sequence, [[token]]], axis=1)
                candidates.append(
                    (
                        extended,
                        score + float(log_probs[token]),
                        cache,
                    )
                )

        selected = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )[:beam_width]
        if step + 1 == max_new_tokens:
            return selected[0][0]

        if mask is not None:
            mask = np.concatenate([mask, np.ones((1, 1), dtype=bool)], axis=1)

        next_beams = []
        for sequence, score, parent_cache in selected:
            if use_cache and _can_extend_cache(parent_cache, model.context_len):
                logits, cache = _extend_cache(model, sequence, parent_cache, mask)
            else:
                logits, cache = _prefill(model, sequence, mask)
                if not use_cache:
                    cache = None
            next_beams.append((sequence, score, logits, cache))
        beams = next_beams

    # The loop returns on its final iteration for every positive token count.
    return beams[0][0]


def _prefill(model, sequence, mask):
    """Infer one strict window and return logits plus its complete KV cache."""
    window = sequence[:, -model.context_len:]
    window_mask = positions = None
    if mask is not None:
        width = window.shape[1]
        window_mask = mask[:, -width:]
        positions = _left_padded_positions(window_mask)
    return model.infer(
        window,
        attention_mask=window_mask,
        position_ids=positions,
    )


def _slice_cache(cache, row):
    """Return one prompt row's read-only view of a batched prefill cache."""
    return [
        {
            "k": entry["k"][row : row + 1],
            "v": entry["v"][row : row + 1],
        }
        for entry in cache
    ]


def _can_extend_cache(cache, context_len):
    return cache is not None and cache[0]["k"].shape[2] < context_len


def _extend_cache(model, sequence, cache, mask):
    """Score the newest token by extending one selected beam's parent cache."""
    cached = cache[0]["k"].shape[2]
    step_mask = step_positions = None
    if mask is not None:
        step_mask = mask[:, -(cached + 1):]
        step_positions = _left_padded_positions(step_mask)[:, -1:]
    return model.infer(
        sequence[:, -1:],
        cache,
        attention_mask=step_mask,
        position_ids=step_positions,
    )
