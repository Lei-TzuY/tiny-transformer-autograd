"""Grouped-query and multi-query causal self-attention.

This module keeps the existing multi-head attention implementation untouched while
adding the modern GQA/MQA projection/cache layout as an independently mergeable
primitive. Query heads remain full-width; fewer key/value heads are projected and
cached, then expanded only for the score/value matmuls.
"""

import numpy as np

import engine.ops as ops
from engine.tensor import Tensor

from .attention import (
    RotaryEmbedding,
    _causal_mask,
    _finite_real,
    _positive_int,
    _prepare_cache,
    _prepare_mask,
    _scaled_dot_product_scores,
    _scaled_dot_product_scores_np,
    _softmax,
)
from .layers import Dropout, Linear
from .module import Module


def _repeat_kv_heads(value: Tensor, repeats: int) -> Tensor:
    """Repeat each KV head contiguously while preserving summed VJPs."""
    if repeats == 1:
        return value
    pieces = []
    for index in range(value.shape[1]):
        head = value[:, index : index + 1, :, :]
        pieces.extend([head] * repeats)
    return ops.concat(pieces, axis=1)


def _repeat_kv_heads_np(value, repeats: int):
    """NumPy equivalent of :func:`_repeat_kv_heads`."""
    return value if repeats == 1 else np.repeat(value, repeats, axis=1)


class GroupedQueryAttention(Module):
    """Causal grouped-query attention with compact K/V projections and cache.

    Parameters
    ----------
    d_model : int
        Input/output model width.
    num_query_heads : int
        Number of query/attention heads.
    num_kv_heads : int or None
        Number of independently projected key/value heads. It must divide
        ``num_query_heads``. ``1`` gives multi-query attention; ``None`` means
        one KV head per query head and is mathematically ordinary MHA.
    dropout : float
        Attention-weight dropout probability.
    rope : RotaryEmbedding or None
        Optional rotary embedding whose dimension equals one query head.

    Notes
    -----
    This pure-NumPy implementation materializes repeated K/V heads for the
    score and value matmuls. The persistent savings therefore come from the
    smaller K/V projection matrices and compact KV cache; this class does not
    claim the kernel-level bandwidth savings of a fused GQA implementation.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.0,
        rope: RotaryEmbedding = None,
    ):
        d_model = _positive_int("d_model", d_model)
        num_query_heads = _positive_int("num_query_heads", num_query_heads)
        if num_kv_heads is None:
            num_kv_heads = num_query_heads
        else:
            num_kv_heads = _positive_int("num_kv_heads", num_kv_heads)
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        dropout = _finite_real(
            "dropout", dropout, lower=0.0, upper=1.0, upper_inclusive=False
        )
        if rope is not None and not isinstance(rope, RotaryEmbedding):
            raise TypeError("rope must be a RotaryEmbedding or None")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.kv_width = num_kv_heads * self.d_k
        self.group_size = num_query_heads // num_kv_heads
        self.scale = self.d_k ** -0.5
        self.rope = rope
        if rope is not None and rope.dim != self.d_k:
            raise ValueError(f"RoPE dimension must equal head dimension {self.d_k}")

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, self.kv_width, bias=False)
        self.W_v = Linear(d_model, self.kv_width, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.attn_drop = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Attend over ``(batch, time, d_model)`` inputs."""
        if not isinstance(x, Tensor):
            raise TypeError("grouped-query attention input must be a Tensor")
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                "grouped-query attention input must have shape "
                f"(batch, time, {self.d_model})"
            )

        B, T, C = x.shape
        Hq, Hkv, d_k = self.num_query_heads, self.num_kv_heads, self.d_k

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = ops.transpose(ops.reshape(Q, (B, T, Hq, d_k)), (0, 2, 1, 3))
        K = ops.transpose(ops.reshape(K, (B, T, Hkv, d_k)), (0, 2, 1, 3))
        V = ops.transpose(ops.reshape(V, (B, T, Hkv, d_k)), (0, 2, 1, 3))

        if self.rope is not None:
            Q = self.rope.rotate(Q)
            K = self.rope.rotate(K)

        K_attn = _repeat_kv_heads(K, self.group_size)
        V_attn = _repeat_kv_heads(V, self.group_size)
        K_T = ops.transpose(K_attn, (0, 1, 3, 2))
        scores = _scaled_dot_product_scores(Q, K_T, self.scale)

        if mask is None:
            mask = Tensor(_causal_mask(T, T, 0))
        else:
            mask = _prepare_mask(mask, scores.shape)
        scores = scores + mask

        weights = self.attn_drop(ops.softmax(scores))
        attn = ops.matmul(weights, V_attn)
        attn = ops.transpose(attn, (0, 2, 1, 3))
        attn = ops.reshape(attn, (B, T, C))
        return self.out_proj(attn)

    def infer(self, x, cache=None, key_bias=None, positions=None):
        """NumPy inference returning logits features and a compact KV cache."""
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                "grouped-query attention input must have shape "
                f"(batch, time, {self.d_model})"
            )
        if not np.issubdtype(x.dtype, np.number) or np.issubdtype(
            x.dtype, np.complexfloating
        ):
            raise TypeError("grouped-query attention input must be real numeric")
        if not np.isfinite(x).all():
            raise ValueError("grouped-query attention input must contain only finite values")

        B, T, C = x.shape
        Hq, Hkv, d_k = self.num_query_heads, self.num_kv_heads, self.d_k
        cache = _prepare_cache(cache, batch=B, width=d_k, heads=Hkv)

        Q = self.W_q.infer(x).reshape(B, T, Hq, d_k).transpose(0, 2, 1, 3)
        K = self.W_k.infer(x).reshape(B, T, Hkv, d_k).transpose(0, 2, 1, 3)
        V = self.W_v.infer(x).reshape(B, T, Hkv, d_k).transpose(0, 2, 1, 3)

        past_len = 0 if cache is None else cache["k"].shape[2]
        if self.rope is not None:
            if positions is None:
                Q = self.rope.rotate_np(Q, offset=past_len)
                K = self.rope.rotate_np(K, offset=past_len)
            else:
                Q = self.rope.rotate_np(Q, positions=positions)
                K = self.rope.rotate_np(K, positions=positions)

        if cache is not None:
            K = np.concatenate([cache["k"], K], axis=2)
            V = np.concatenate([cache["v"], V], axis=2)

        # Preserve the compact cache. Repetition is only an execution-time view
        # of the mapping from query heads to their shared K/V head.
        compact_cache = {"k": K, "v": V}
        K_attn = _repeat_kv_heads_np(K, self.group_size)
        V_attn = _repeat_kv_heads_np(V, self.group_size)

        scores = _scaled_dot_product_scores_np(
            Q, K_attn.transpose(0, 1, 3, 2), self.scale
        )
        scores += _causal_mask(T, K.shape[2], past_len)
        if key_bias is not None:
            key_bias = _prepare_mask(
                key_bias,
                scores.shape,
                as_tensor=False,
                name="attention key_bias",
            )
            scores = scores + key_bias
        weights = _softmax(scores)
        attn = weights @ V_attn
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj.infer(attn), compact_cache

    def __repr__(self):
        return (
            "GroupedQueryAttention("
            f"d_model={self.d_model}, query_heads={self.num_query_heads}, "
            f"kv_heads={self.num_kv_heads}, dropout={self.attn_drop.p})"
        )
