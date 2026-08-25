"""Beam search helpers with explicit left-padding support.

``GPT.generate_beam`` predates ragged prompt masks and intentionally remains a
small unmasked reference path. ``beam_generate`` adds the missing masked form
without changing that public method's behaviour. It still decodes one sequence
at a time and re-prefills the strict cropped window for every beam candidate.
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
):
    """Return the highest-scoring sequence from strict-window beam search.

    ``attention_mask`` follows generation semantics: one left-padded row where
    0 marks padding and 1 marks real prompt tokens. Generated tokens are always
    real. The returned array preserves the original padding columns.

    Unlike the incremental sampling path, beam search deliberately performs a
    fresh inference for each candidate sequence. When a sequence exceeds
    ``context_len``, only its newest window is visible and positions are
    renumbered from zero, exactly matching strict ``GPT.generate`` semantics.
    """
    if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not isinstance(beam_width, (int, np.integer)) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    if idx.shape[0] != 1:
        raise ValueError("beam search currently supports batch size 1")

    mask = None
    if attention_mask is not None:
        mask = model._validate_generation_mask(attention_mask, idx.shape).copy()

    beams = [(idx, 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        for sequence, score in beams:
            window = sequence[:, -model.context_len:]
            window_mask = positions = None
            if mask is not None:
                width = window.shape[1]
                window_mask = mask[:, -width:]
                positions = _left_padded_positions(window_mask)

            logits, _ = model.infer(
                window,
                attention_mask=window_mask,
                position_ids=positions,
            )
            log_probs = _log_softmax(logits[0, -1] / temperature)
            best = np.argsort(log_probs)[-beam_width:]
            for token in best:
                extended = np.concatenate([sequence, [[token]]], axis=1)
                candidates.append((extended, score + float(log_probs[token])))

        beams = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )[:beam_width]
        if mask is not None:
            mask = np.concatenate([mask, np.ones((1, 1), dtype=bool)], axis=1)

    return beams[0][0]
