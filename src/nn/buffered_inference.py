"""Attention inference that appends into :class:`KVCacheBuffer` in place.

The historical ``SelfAttention.infer`` and ``MultiHeadAttention.infer`` APIs use a
``{"k": ..., "v": ...}`` mapping and grow an existing cache with ``np.concatenate``.
That representation remains useful and unchanged.  This module provides an explicit
opt-in path for callers that want fixed-capacity cache storage instead.

``infer_with_kv_buffer`` deliberately reuses the attention module's existing numerical
helpers (overflow-safe score computation, masking, softmax, RoPE, and projection
implementations).  The only changed concern is cache ownership/growth: new K/V chunks
are appended to a caller-owned ``KVCacheBuffer`` and the returned cache mapping is a
read-only live view of that buffer.
"""

import numpy as np

from .attention import (
    MultiHeadAttention,
    SelfAttention,
    _causal_mask,
    _prepare_cache,
    _prepare_mask,
    _scaled_dot_product_scores_np,
    _softmax,
)
from .kv_cache import KVCacheBuffer


def infer_with_kv_buffer(
    attention,
    x,
    buffer,
    *,
    key_bias=None,
    positions=None,
):
    """Run NumPy attention inference while growing ``buffer`` without concatenation.

    Parameters
    ----------
    attention:
        A ``SelfAttention`` or ``MultiHeadAttention`` instance.
    x:
        The same NumPy input accepted by the selected module's ordinary ``infer``.
    buffer:
        Caller-owned ``KVCacheBuffer``.  Its complete lock is held for the call, so
        two helper-managed inferences cannot observe the same buffer half-updated.
    key_bias:
        Same additive per-key bias accepted by the ordinary inference APIs.
    positions:
        Explicit RoPE positions for ``MultiHeadAttention``.  Single-head attention
        has no positions argument and therefore rejects a non-``None`` value.

    Returns
    -------
    (output, cache_view)
        ``cache_view`` is the read-only live ``{"k", "v"}`` mapping returned by the
        buffer.  The buffer itself remains owned by the caller.

    Notes
    -----
    The helper is transactional with respect to the buffer's *visible* state.  For an
    already initialized buffer, a failure after append truncates the logical length to
    its entry value.  For the very first append, all attention computation is completed
    before storage is allocated/published, so a failure leaves the buffer uninitialized.
    """
    if not isinstance(buffer, KVCacheBuffer):
        raise TypeError("buffer must be a KVCacheBuffer")
    if isinstance(attention, SelfAttention):
        if positions is not None:
            raise TypeError("positions are only supported by MultiHeadAttention")
        return _infer_self_attention(attention, x, buffer, key_bias=key_bias)
    if isinstance(attention, MultiHeadAttention):
        return _infer_multi_head_attention(
            attention,
            x,
            buffer,
            key_bias=key_bias,
            positions=positions,
        )
    raise TypeError("attention must be SelfAttention or MultiHeadAttention")


def _infer_self_attention(attention, x, buffer, *, key_bias):
    # KVCacheBuffer owns an instance RLock.  Holding it across the complete inference
    # linearizes past-length selection, RoPE/mask semantics, append, and rollback.
    with buffer._lock:
        existing = _prepared_buffer_view(
            buffer,
            batch=x.shape[0],
            width=attention.d_model,
        )
        past_len = 0 if existing is None else existing["k"].shape[1]
        query_len = x.shape[1]
        _preflight_capacity(buffer, query_len)

        Q = attention.W_q.infer(x)
        K_new = attention.W_k.infer(x)
        V_new = attention.W_v.infer(x)

        if existing is None and not buffer.initialized:
            # On the first ever append the complete numerical path can be evaluated
            # directly against the new chunk.  Only publish storage after it succeeds.
            output = _finish_self_attention(
                attention,
                Q,
                K_new,
                V_new,
                past_len=0,
                key_bias=key_bias,
            )
            live = buffer.append(K_new, V_new)
            return output, live

        entry_len = past_len
        appended = False
        try:
            live = buffer.append(K_new, V_new)
            appended = True
            output = _finish_self_attention(
                attention,
                Q,
                live["k"],
                live["v"],
                past_len=entry_len,
                key_bias=key_bias,
            )
            return output, live
        except BaseException:
            if appended:
                _rollback_visible_length(buffer, entry_len)
            raise


def _finish_self_attention(attention, query, key, value, *, past_len, key_bias):
    scores = _scaled_dot_product_scores_np(
        query,
        np.swapaxes(key, -1, -2),
        attention.scale,
    )
    scores += _causal_mask(query.shape[1], key.shape[1], past_len)
    if key_bias is not None:
        key_bias = _prepare_mask(
            key_bias,
            scores.shape,
            as_tensor=False,
            name="attention key_bias",
        )
        scores = scores + key_bias
    weights = _softmax(scores)
    return attention.out_proj.infer(weights @ value)


def _infer_multi_head_attention(attention, x, buffer, *, key_bias, positions):
    with buffer._lock:
        B, T, C = x.shape
        H, d_k = attention.num_heads, attention.d_k
        existing = _prepared_buffer_view(
            buffer,
            batch=B,
            width=d_k,
            heads=H,
        )
        past_len = 0 if existing is None else existing["k"].shape[2]
        _preflight_capacity(buffer, T)

        Q = attention.W_q.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K_new = attention.W_k.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V_new = attention.W_v.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        if attention.rope is not None:
            if positions is None:
                Q = attention.rope.rotate_np(Q, offset=past_len)
                K_new = attention.rope.rotate_np(K_new, offset=past_len)
            else:
                Q = attention.rope.rotate_np(Q, positions=positions)
                K_new = attention.rope.rotate_np(K_new, positions=positions)

        if existing is None and not buffer.initialized:
            output = _finish_multi_head_attention(
                attention,
                Q,
                K_new,
                V_new,
                past_len=0,
                key_bias=key_bias,
                batch=B,
                query_len=T,
                model_width=C,
            )
            live = buffer.append(K_new, V_new)
            return output, live

        entry_len = past_len
        appended = False
        try:
            live = buffer.append(K_new, V_new)
            appended = True
            output = _finish_multi_head_attention(
                attention,
                Q,
                live["k"],
                live["v"],
                past_len=entry_len,
                key_bias=key_bias,
                batch=B,
                query_len=T,
                model_width=C,
            )
            return output, live
        except BaseException:
            if appended:
                _rollback_visible_length(buffer, entry_len)
            raise


def _finish_multi_head_attention(
    attention,
    query,
    key,
    value,
    *,
    past_len,
    key_bias,
    batch,
    query_len,
    model_width,
):
    scores = _scaled_dot_product_scores_np(
        query,
        key.transpose(0, 1, 3, 2),
        attention.scale,
    )
    scores += _causal_mask(query_len, key.shape[2], past_len)
    if key_bias is not None:
        key_bias = _prepare_mask(
            key_bias,
            scores.shape,
            as_tensor=False,
            name="attention key_bias",
        )
        scores = scores + key_bias
    weights = _softmax(scores)
    attended = weights @ value
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch,
        query_len,
        model_width,
    )
    return attention.out_proj.infer(attended)


def _prepared_buffer_view(buffer, *, batch, width, heads=None):
    if not buffer.initialized:
        return None
    return _prepare_cache(
        buffer.view(),
        batch=batch,
        width=width,
        heads=heads,
    )


def _preflight_capacity(buffer, query_len):
    if query_len > buffer.remaining:
        raise OverflowError(
            f"cache capacity {buffer.capacity} exceeded by append ending at "
            f"{buffer.length + query_len}"
        )


def _rollback_visible_length(buffer, entry_len):
    try:
        buffer.truncate(entry_len)
    except BaseException as exc:
        raise RuntimeError("attention KV cache rollback failed") from exc
