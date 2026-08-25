"""Fast autoregressive generation with explicit streaming-cache semantics.

The regular :meth:`GPT.generate` implements a *strict* sliding window: when the
context fills it re-prefills the cropped window so every surviving hidden state
is recomputed as if the dropped token had never been present.

``stream_generate`` chooses a different, common deployment policy.  Once the
RoPE cache fills, it drops the oldest K/V slot, rebases surviving keys by one
position, and keeps decoding incrementally.  This is exact for a one-block
Transformer.  With deeper models, higher-layer cached states can retain
indirect information about tokens that have left the window, so the result is
intentionally not claimed to equal a strict re-prefill.

Keeping the policy in a separate function makes that semantic choice explicit
instead of silently changing ``GPT.generate`` in the name of performance.
"""

import numpy as np

from .transformer import _left_padded_positions, _sample, _validate_selection_logits


def stream_generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    strategy="sample",
    attention_mask=None,
):
    """Generate with a bounded, shifted RoPE KV cache.

    Parameters mirror the sampling subset of ``GPT.generate``.  ``model`` must
    use ``pos_encoding='rope'``. Beam search is deliberately absent because
    this helper represents one linear streaming cache, not a beam cache tree.

    The returned token array keeps any left padding from the prompt, matching
    ``GPT.generate``. ``max_new_tokens=0`` returns a copy of the validated
    prompt without running inference.
    """
    if getattr(model, "rope", None) is None:
        raise ValueError("stream_generate requires a model with pos_encoding='rope'")
    if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if strategy not in {"sample", "greedy"}:
        raise ValueError("strategy must be 'sample' or 'greedy'")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    idx = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    keep = None
    if attention_mask is not None:
        keep = model._validate_generation_mask(attention_mask, idx.shape).copy()
    if max_new_tokens == 0:
        return idx

    # Match strict generation's first view of the prompt: only the newest
    # context_len slots are visible, and surviving real tokens restart at 0.
    window = idx[:, -model.context_len:]
    cache_keep = None
    positions = None
    if keep is not None:
        cache_keep = keep[:, -window.shape[1]:]
        positions = _left_padded_positions(cache_keep)
    logits, cache = model.infer(
        window,
        attention_mask=cache_keep,
        position_ids=positions,
    )

    for step in range(max_new_tokens):
        logits_last = _validate_selection_logits(
            logits[:, -1, :], "generation logits"
        )
        if strategy == "greedy":
            next_token = np.argmax(logits_last, axis=-1)
        else:
            next_token = np.array(
                [
                    _sample(
                        logit,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    for logit in logits_last
                ]
            )
        idx = np.concatenate([idx, next_token[:, None]], axis=1)
        if keep is not None:
            keep = np.concatenate(
                [keep, np.ones((keep.shape[0], 1), dtype=bool)], axis=1
            )

        if step + 1 == max_new_tokens:
            break

        if cache[0]["k"].shape[2] >= model.context_len:
            dropped_real = (
                np.ones(idx.shape[0], dtype=bool)
                if cache_keep is None
                else cache_keep[:, 0]
            )
            cache = _drop_oldest_and_rebase(cache, model, dropped_real)
            if cache_keep is not None:
                cache_keep = cache_keep[:, 1:]

        step_keep = step_positions = None
        if cache_keep is not None:
            step_keep = np.concatenate(
                [
                    cache_keep,
                    np.ones((cache_keep.shape[0], 1), dtype=bool),
                ],
                axis=1,
            )
            # Positions are derived from the keys that actually remain in the
            # cache. Dropping a padding slot leaves real positions unchanged;
            # dropping a real slot renumbers every surviving real key by -1.
            step_positions = _left_padded_positions(step_keep)[:, -1:]

        logits, cache = model.infer(
            idx[:, -1:],
            cache,
            attention_mask=step_keep,
            position_ids=step_positions,
        )
        if cache_keep is not None:
            cache_keep = step_keep

    return idx


def _drop_oldest_and_rebase(cache, model, dropped_real):
    """Drop one KV slot and rebase RoPE keys only for rows that lost a token."""
    dropped_real = np.asarray(dropped_real, dtype=bool)
    shifted = []
    for entry in cache:
        key = entry["k"][:, :, 1:, :].copy()
        value = entry["v"][:, :, 1:, :].copy()
        if key.shape[2] and np.any(dropped_real):
            # R(-1)x = cos(1)x - sin(1)Jx.  ``rotate_half`` is J.
            half = key.shape[-1] // 2
            rotate_half = np.concatenate(
                [-key[..., half:], key[..., :half]], axis=-1
            )
            rebased = key * model.rope.cos[1] - rotate_half * model.rope.sin[1]
            key = np.where(
                dropped_real[:, None, None, None],
                rebased,
                key,
            )
        shifted.append({"k": key, "v": value})
    return shifted
