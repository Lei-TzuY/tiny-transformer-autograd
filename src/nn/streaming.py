"""Fast autoregressive generation with explicit streaming-cache semantics.

The regular :meth:`GPT.generate` implements a *strict* sliding window: when the
context fills it re-prefills the cropped window so every surviving hidden state
is recomputed as if the dropped token had never been present.

``stream_generate`` chooses a different, common deployment policy. Once the
RoPE cache fills, it drops the oldest K/V slot, rebases surviving keys by one
position, and keeps decoding incrementally. This is exact for a one-block
Transformer. With deeper models, higher-layer cached states can retain indirect
information about tokens that have left the window, so the result is
intentionally not claimed to equal a strict re-prefill.

``stream_generate_iter`` exposes the same state machine one generated token at
a time. Consumers can render output immediately or stop iteration without
running inference for tokens they no longer need. ``stream_generate`` simply
collects that iterator, keeping one source of decoding semantics.

An optional ``stop_token_id`` terminates generation once every batch row has
emitted that token. Rows that finish earlier enter an absorbing stop state, so
the rectangular batch output remains well-defined while active rows continue.
"""

import numpy as np

from .transformer import (
    _left_padded_positions,
    _sample,
    _validate_non_negative_int,
    _validate_sampling_options,
    _validate_selection_logits,
)


def stream_generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    strategy="sample",
    attention_mask=None,
    stop_token_id=None,
):
    """Generate with a bounded, shifted RoPE KV cache.

    Parameters mirror the sampling subset of ``GPT.generate``. ``model`` must
    use ``pos_encoding='rope'``. Beam search is deliberately absent because
    this helper represents one linear streaming cache, not a beam cache tree.

    ``stop_token_id`` applies only to newly generated tokens. A row that emits
    it keeps emitting that same token while other rows continue; generation
    stops as soon as every row has finished. The stop token itself is included
    in the returned array.

    The returned token array keeps any left padding from the prompt, matching
    ``GPT.generate``. ``max_new_tokens=0`` returns a copy of the validated
    prompt without running inference.
    """
    prompt, iterator = _make_stream_iterator(
        model,
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
        attention_mask=attention_mask,
        stop_token_id=stop_token_id,
    )
    generated = list(iterator)
    if not generated:
        return prompt
    return np.concatenate([prompt, np.stack(generated, axis=1)], axis=1)


def stream_generate_iter(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    strategy="sample",
    attention_mask=None,
    stop_token_id=None,
):
    """Return an iterator yielding one generated token per batch row per step.

    Validation is eager, but inference is lazy: merely creating the iterator
    does not run the model. Each ``next()`` advances decoding only far enough to
    select one token and yields an independent ``int64`` array with shape
    ``(batch,)``. Stopping or closing the iterator therefore performs no future
    inference. Mutating a yielded array cannot change subsequent decoding.

    With ``stop_token_id`` set, the iterator ends when every row has emitted the
    stop token. Finished rows stay at that token while unfinished rows continue.
    Fully consuming the iterator is exactly the generated suffix returned by
    :func:`stream_generate` for the same model, inputs, options, and RNG state.
    """
    _, iterator = _make_stream_iterator(
        model,
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
        attention_mask=attention_mask,
        stop_token_id=stop_token_id,
    )
    return iterator


def _validate_stop_token_id(model, stop_token_id):
    """Normalize one optional vocabulary id used as an absorbing stop token."""
    if stop_token_id is None:
        return None
    if isinstance(stop_token_id, (bool, np.bool_)) or not isinstance(
        stop_token_id, (int, np.integer)
    ):
        raise TypeError("stop_token_id must be an integer or None")
    stop_token_id = int(stop_token_id)
    if stop_token_id < 0 or stop_token_id >= model.vocab_size:
        raise ValueError(
            f"stop_token_id must be in [0, {model.vocab_size - 1}]"
        )
    return stop_token_id


def _make_stream_iterator(
    model,
    idx,
    max_new_tokens,
    *,
    temperature,
    top_k,
    top_p,
    strategy,
    attention_mask,
    stop_token_id,
):
    """Validate one request eagerly and build its lazy token-step iterator."""
    if getattr(model, "rope", None) is None:
        raise ValueError("stream_generate requires a model with pos_encoding='rope'")
    max_new_tokens = _validate_non_negative_int(max_new_tokens, "max_new_tokens")
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string")
    if strategy not in {"sample", "greedy"}:
        raise ValueError("strategy must be 'sample' or 'greedy'")
    temperature, top_k, top_p = _validate_sampling_options(
        temperature, top_k, top_p
    )
    stop_token_id = _validate_stop_token_id(model, stop_token_id)

    prompt = np.array(model._validate_token_batch(idx), dtype=np.int64, copy=True)
    keep = None
    if attention_mask is not None:
        keep = model._validate_generation_mask(attention_mask, prompt.shape).copy()

    iterator = _stream_generation_steps(
        model,
        prompt,
        keep,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        strategy,
        stop_token_id,
    )
    return prompt, iterator


def _select_next_tokens(
    logits_last,
    strategy,
    temperature,
    top_k,
    top_p,
    stop_token_id,
    finished,
):
    """Select one token per row without consuming RNG for finished rows."""
    batch = logits_last.shape[0]
    if strategy == "greedy":
        next_token = np.argmax(logits_last, axis=-1).astype(np.int64, copy=False)
    elif finished is None:
        next_token = np.array(
            [
                _sample(
                    logit,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                for logit in logits_last
            ],
            dtype=np.int64,
        )
    else:
        next_token = np.full(batch, stop_token_id, dtype=np.int64)
        for row, logit in enumerate(logits_last):
            if not finished[row]:
                next_token[row] = _sample(
                    logit,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

    if finished is not None and np.any(finished):
        next_token = np.where(finished, stop_token_id, next_token).astype(
            np.int64, copy=False
        )
    return next_token


def _stream_generation_steps(
    model,
    prompt,
    keep,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    strategy,
    stop_token_id,
):
    """Yield selected token ids while owning all shifted-cache state."""
    if max_new_tokens == 0:
        return

    # Match strict generation's first view of the prompt: only the newest
    # context_len slots are visible, and surviving real tokens restart at 0.
    window = prompt[:, -model.context_len:]
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
    finished = (
        None
        if stop_token_id is None
        else np.zeros(prompt.shape[0], dtype=bool)
    )

    for step in range(max_new_tokens):
        logits_last = _validate_selection_logits(
            logits[:, -1, :], "generation logits"
        )
        next_token = _select_next_tokens(
            logits_last,
            strategy,
            temperature,
            top_k,
            top_p,
            stop_token_id,
            finished,
        )
        if finished is not None:
            finished |= next_token == stop_token_id

        # Yield a copy so consumer mutation cannot rewrite the state used by the
        # next cache extension. No next-step inference occurs until iteration resumes.
        yield np.array(next_token, dtype=np.int64, copy=True)
        if finished is not None and np.all(finished):
            break
        if step + 1 == max_new_tokens:
            break

        if cache[0]["k"].shape[2] >= model.context_len:
            dropped_real = (
                np.ones(prompt.shape[0], dtype=bool)
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
            next_token[:, None],
            cache,
            attention_mask=step_keep,
            position_ids=step_positions,
        )
        if cache_keep is not None:
            cache_keep = step_keep


def _drop_oldest_and_rebase(cache, model, dropped_real):
    """Drop one KV slot and rebase RoPE keys only for rows that lost a token."""
    dropped_real = np.asarray(dropped_real, dtype=bool)
    shifted = []
    for entry in cache:
        key = entry["k"][:, :, 1:, :].copy()
        value = entry["v"][:, :, 1:, :].copy()
        if key.shape[2] and np.any(dropped_real):
            # R(-1)x = cos(1)x - sin(1)Jx. ``rotate_half`` is J.
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
