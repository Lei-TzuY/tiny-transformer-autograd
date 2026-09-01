"""Beam search helpers with explicit left-padding support.

``GPT.generate_beam`` predates ragged prompt masks and intentionally remains a
small unmasked reference path. ``beam_generate`` adds masked and batched forms
without changing that method, and can keep one KV cache per selected beam until
strict sliding-window semantics require a re-prefill.
"""

import numpy as np

from .transformer import (
    _left_padded_positions,
    _log_softmax,
    _temperature_scale_logits,
    _validate_non_negative_int,
    _validate_positive_finite_real,
    _validate_positive_int,
)


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

    Beam scores and candidate selection stay independent per prompt row, but
    inference is tensorized wherever the rows share compatible shapes. Prompt
    prefill runs once for the whole request. On later steps, every row selects
    its own best children, then all selected children across all prompt rows are
    flattened into one inference batch. Immutable KV snapshots make that cache
    batching safe even when siblings share the same parent state.

    Equal-scoring token candidates use descending token id as a deterministic
    secondary key. This makes exact ties independent of NumPy's sorting-kind
    implementation details and, importantly, independent of ``beam_width``.

    When a cache fills ``context_len``, all selected children are re-prefilled
    together from their cropped strict windows so positions are renumbered and
    out-of-window tokens are genuinely forgotten.
    """
    max_new_tokens = _validate_non_negative_int(max_new_tokens, "max_new_tokens")
    beam_width = _validate_positive_int(beam_width, "beam_width")
    temperature = _validate_positive_finite_real(temperature, "temperature")
    if not isinstance(use_cache, (bool, np.bool_)):
        raise TypeError("use_cache must be boolean")
    use_cache = bool(use_cache)

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    mask = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape).copy()
    if max_new_tokens == 0:
        return idx

    logits, cache = _prefill(model, idx, mask)

    beam_groups = []
    row_masks = []
    for row in range(idx.shape[0]):
        row_cache = _slice_cache(cache, row) if use_cache else None
        beam_groups.append(
            [(idx[row : row + 1], 0.0, logits[row : row + 1], row_cache)]
        )
        row_masks.append(None if mask is None else mask[row : row + 1])

    for step in range(max_new_tokens):
        selected_groups = [
            _select_children(beams, beam_width, temperature)
            for beams in beam_groups
        ]
        if step + 1 == max_new_tokens:
            return np.concatenate(
                [selected[0][0] for selected in selected_groups],
                axis=0,
            )

        if mask is not None:
            row_masks = [
                np.concatenate(
                    [row_mask, np.ones((1, 1), dtype=bool)],
                    axis=1,
                )
                for row_mask in row_masks
            ]

        beam_groups = _advance_selected_groups(
            model,
            selected_groups,
            row_masks,
            use_cache,
        )

    return idx


def _rank_beam_tokens(log_probs, beam_width):
    """Rank token ids by score descending, then token id descending."""
    token_ids = np.arange(log_probs.shape[0], dtype=np.int64)
    order = np.lexsort((-token_ids, -log_probs))
    return order[:beam_width]


def _select_children(beams, beam_width, temperature):
    """Select one prompt row's highest-scoring next beam candidates."""
    candidates = []
    for sequence, score, logits, cache in beams:
        scaled = _temperature_scale_logits(logits[0, -1], temperature)
        log_probs = _log_softmax(scaled)
        best = _rank_beam_tokens(log_probs, beam_width)
        for token in best:
            extended = np.concatenate([sequence, [[token]]], axis=1)
            candidates.append(
                (
                    extended,
                    score + float(log_probs[token]),
                    cache,
                )
            )
    return sorted(
        candidates,
        key=lambda item: item[1],
        reverse=True,
    )[:beam_width]


def _advance_selected(model, selected, mask, use_cache):
    """Compatibility wrapper for advancing one prompt row's selected beams."""
    return _advance_selected_groups(model, [selected], [mask], use_cache)[0]


def _advance_selected_groups(model, selected_groups, row_masks, use_cache):
    """Score selected children from every prompt row in one inference batch."""
    if len(selected_groups) != len(row_masks):
        raise ValueError("selected beam groups and masks must have equal length")

    flattened = [item for group in selected_groups for item in group]
    if not flattened:
        raise ValueError("cannot advance an empty beam selection")

    sequences = np.concatenate([sequence for sequence, _, _ in flattened], axis=0)
    parent_caches = [parent_cache for _, _, parent_cache in flattened]

    if row_masks[0] is None:
        if any(row_mask is not None for row_mask in row_masks):
            raise ValueError("beam row masks must be either all present or all absent")
        batch_mask = None
    else:
        if any(row_mask is None for row_mask in row_masks):
            raise ValueError("beam row masks must be either all present or all absent")
        batch_mask = np.concatenate(
            [
                np.repeat(row_mask, len(group), axis=0)
                for group, row_mask in zip(selected_groups, row_masks)
            ],
            axis=0,
        )

    if use_cache and all(
        _can_extend_cache(parent_cache, model.context_len)
        for parent_cache in parent_caches
    ):
        batched_parent = _stack_cache(parent_caches)
        logits, cache = _extend_cache(model, sequences, batched_parent, batch_mask)
    else:
        logits, cache = _prefill(model, sequences, batch_mask)
        if not use_cache:
            cache = None

    next_groups = []
    flat_row = 0
    for group in selected_groups:
        next_group = []
        for sequence, score, _ in group:
            row_cache = None if cache is None else _slice_cache(cache, flat_row)
            next_group.append(
                (
                    sequence,
                    score,
                    logits[flat_row : flat_row + 1],
                    row_cache,
                )
            )
            flat_row += 1
        next_groups.append(next_group)
    return next_groups


def _freeze_cache(cache):
    """Make a beam-owned cache immutable in place and return it."""
    if cache is None:
        return None
    for entry in cache:
        entry["k"].flags.writeable = False
        entry["v"].flags.writeable = False
    return cache


def _prefill(model, sequence, mask):
    """Infer one strict window and return logits plus its immutable KV cache."""
    window = sequence[:, -model.context_len:]
    window_mask = positions = None
    if mask is not None:
        width = window.shape[1]
        window_mask = mask[:, -width:]
        positions = _left_padded_positions(window_mask)
    logits, cache = model.infer(
        window,
        attention_mask=window_mask,
        position_ids=positions,
    )
    return logits, _freeze_cache(cache)


def _slice_cache(cache, row):
    """Return one row's zero-copy immutable view of a batched cache."""
    return _freeze_cache(
        [
            {
                "k": entry["k"][row : row + 1],
                "v": entry["v"][row : row + 1],
            }
            for entry in cache
        ]
    )


def _stack_cache(caches):
    """Batch immutable per-beam caches for one shared inference call."""
    if not caches:
        raise ValueError("cannot batch an empty cache collection")
    layers = len(caches[0])
    if any(len(cache) != layers for cache in caches):
        raise ValueError("all beam caches must contain the same number of layers")
    return _freeze_cache(
        [
            {
                "k": np.concatenate([cache[layer]["k"] for cache in caches], axis=0),
                "v": np.concatenate([cache[layer]["v"] for cache in caches], axis=0),
            }
            for layer in range(layers)
        ]
    )


def _can_extend_cache(cache, context_len):
    return cache is not None and cache[0]["k"].shape[2] < context_len


def _extend_cache(model, sequence, cache, mask):
    """Score newest tokens from one immutable batched parent cache."""
    cached = cache[0]["k"].shape[2]
    step_mask = step_positions = None
    if mask is not None:
        step_mask = mask[:, -(cached + 1):]
        step_positions = _left_padded_positions(step_mask)[:, -1:]
    logits, child_cache = model.infer(
        sequence[:, -1:],
        cache,
        attention_mask=step_mask,
        position_ids=step_positions,
    )
    return logits, _freeze_cache(child_cache)
