"""
attention.py — Scaled dot-product self-attention (single-head and multi-head).

Scaled dot-product attention
-----------------------------
Given queries Q, keys K, values V all of shape (..., T, d_k):

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) · V

The causal mask is an upper-triangular matrix of −∞ (passed in as a Tensor
with requires_grad=False so gradients stop there).

Multi-head variant
------------------
Split d_model into H independent heads of size d_k = d_model / H, run
attention in parallel, then concatenate and project back.

Input/output shapes
-------------------
x       : (B, T, d_model)
mask    : (T, T) — broadcasts over batch and head dims automatically
output  : (B, T, d_model)

Custom masks
------------
``forward(x, mask)`` accepts any additive bias that broadcasts to the score
shape — (T, T), (B, T, T), or (B, H, T, T) for multi-head.  Entries are added
to the scores before the softmax, so 0.0 keeps a key and −∞ removes it.  NaN
and +∞ are rejected: they cannot express "masked" and would silently turn the
whole batch into NaN.

Fully masked query rows
-----------------------
A row whose every key is −∞ (a padded query position, or an over-restrictive
custom mask) has no distribution to normalise.  By the convention defined in
``ops.softmax``, its attention weights are **all zero**, so:

  - the row's context vector is exactly zero and the layer output for that
    position is just ``out_proj``'s bias — a constant, not a real prediction;
  - no gradient flows from that position back to Q, K, or V.

Nothing becomes NaN, but that position's output is meaningless by
construction: exclude it from the loss instead of trusting it.
"""

import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from .module import Module
from .layers import Linear, Dropout


class RotaryEmbedding:
    """
    Rotary position embedding (RoFormer, Su et al. 2021; used by Llama).

    Instead of adding a learned position vector to the token embedding, RoPE
    rotates each (q, k) head vector by a position-dependent angle.  Pairs of
    channels (i, i + d/2) form 2-D planes; position p rotates plane i by
    p · θᵢ where θᵢ = base^(−2i/d).  Attention scores q·k then depend only on
    the *relative* offset between positions — a property learned absolute
    embeddings lack.

    The cos/sin tables are precomputed constants (no trainable parameters),
    so this is deliberately not a Module.
    """

    def __init__(self, dim: int, max_pos: int, base: float = 10000.0):
        if dim <= 0 or max_pos <= 0:
            raise ValueError("RoPE dimension and max_pos must be positive")
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        if base <= 0:
            raise ValueError("RoPE base must be positive")
        inv_freq = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
        angles = np.outer(np.arange(max_pos, dtype=np.float64), inv_freq)
        emb = np.concatenate([angles, angles], axis=-1)  # (max_pos, dim)
        self.dim = dim
        self.max_pos = max_pos
        self.base = base
        self.cos = np.cos(emb)
        self.sin = np.sin(emb)

    def rotate(self, x: Tensor, offset: int = 0) -> Tensor:
        """Rotate a (..., T, dim) Tensor in the autograd graph."""
        self._validate_input(x.shape, offset)
        T = x.shape[-2]
        cos = Tensor(self.cos[offset : offset + T])
        sin = Tensor(self.sin[offset : offset + T])
        return x * cos + _rotate_half(x) * sin

    def rotate_np(self, x, offset: int = 0, positions=None):
        """
        NumPy-only rotation for the inference path.

        ``positions`` supplies an explicit absolute position per element, shaped
        to broadcast against ``x.shape[:-1]``.  It is what makes left-padded
        batched generation work: rows of one batch then sit at different
        absolute positions even though they share a slot index.  Without it
        every row is assumed to start at ``offset``.
        """
        x = np.asarray(x)
        if positions is None:
            self._validate_input(x.shape, offset)
            T = x.shape[-2]
            cos = self.cos[offset : offset + T]
            sin = self.sin[offset : offset + T]
        else:
            cos, sin = self._gather_positions(x.shape, positions)
        return x * cos + _rotate_half_np(x) * sin

    def _validate_input(self, shape, offset):
        if len(shape) < 2 or shape[-1] != self.dim:
            raise ValueError(f"RoPE input must end in dimension {self.dim}")
        if not isinstance(offset, (int, np.integer)) or offset < 0:
            raise ValueError("RoPE offset must be a non-negative integer")
        if offset + shape[-2] > self.max_pos:
            raise ValueError("RoPE input positions exceed max_pos")

    def _gather_positions(self, shape, positions):
        """Look up the cos/sin rows for arbitrary per-element positions."""
        if len(shape) < 2 or shape[-1] != self.dim:
            raise ValueError(f"RoPE input must end in dimension {self.dim}")
        positions = np.asarray(positions)
        if not np.issubdtype(positions.dtype, np.integer):
            raise TypeError("RoPE positions must be integers")
        if positions.size == 0:
            raise ValueError("RoPE positions must not be empty")
        if positions.min() < 0 or positions.max() >= self.max_pos:
            raise ValueError(f"RoPE positions must be in [0, {self.max_pos})")
        leading = tuple(shape[:-1])
        try:
            broadcast = np.broadcast_shapes(positions.shape, leading)
        except ValueError as exc:
            raise ValueError(
                f"RoPE positions {positions.shape} do not broadcast to the "
                f"input's leading axes {leading}"
            ) from exc
        if broadcast != leading:
            raise ValueError(
                f"RoPE positions {positions.shape} are larger than the input's "
                f"leading axes {leading}"
            )
        # Indexing appends the rotation dimension, so the result already
        # broadcasts against x.
        return self.cos[positions], self.sin[positions]


def _rotate_half(x: Tensor) -> Tensor:
    """[x₁, x₂] → [−x₂, x₁] along the last axis (graph-tracked)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concat([-x2, x1], axis=-1)


def _rotate_half_np(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class SelfAttention(Module):
    """Single-head causal self-attention."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.scale = d_model ** -0.5

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.attn_drop = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Attend over (B, T, d_model) inputs.

        ``mask=None`` uses the causal mask, matching ``infer()``. A custom mask
        is an additive bias broadcastable to (B, T, T); a query row that is
        entirely −∞ produces zero attention weights (see the module docstring).
        """
        # x: (B, T, d_model)
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # scores: (B, T, T)
        K_T = ops.transpose(K, (0, 2, 1))          # (B, d_model, T)
        scores = ops.matmul(Q, K_T) * self.scale    # (B, T, T)

        if mask is None:
            mask = Tensor(_causal_mask(x.shape[1], x.shape[1], 0))
        else:
            mask = _prepare_mask(mask, scores.shape)
        scores = scores + mask  # (T, T) broadcasts to (B, T, T)

        weights = ops.softmax(scores)               # (B, T, T)
        weights = self.attn_drop(weights)

        out = ops.matmul(weights, V)                # (B, T, d_model)
        return self.out_proj(out)

    def infer(self, x, cache=None, key_bias=None):
        """
        NumPy-only inference with an optional key/value cache.

        ``key_bias`` is an additive per-key bias broadcastable to the
        (batch, query, key) scores — used to hide padded keys.
        """
        Q = self.W_q.infer(x)
        K = self.W_k.infer(x)
        V = self.W_v.infer(x)
        past_len = 0 if cache is None else cache["k"].shape[1]
        if cache is not None:
            K = np.concatenate([cache["k"], K], axis=1)
            V = np.concatenate([cache["v"], V], axis=1)
        scores = Q @ np.swapaxes(K, -1, -2) * self.scale
        scores += _causal_mask(x.shape[1], K.shape[1], past_len)
        if key_bias is not None:
            scores = scores + key_bias
        weights = _softmax(scores)
        return self.out_proj.infer(weights @ V), {"k": K, "v": V}


class MultiHeadAttention(Module):
    """
    Multi-head causal self-attention (GPT-style, pre-norm variant).

    Uses separate W_q, W_k, W_v projections for clarity, then merges heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope: RotaryEmbedding = None,
    ):
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5
        self.rope = rope
        if rope is not None and rope.dim != self.d_k:
            raise ValueError(f"RoPE dimension must equal head dimension {self.d_k}")

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.attn_drop = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Attend over (B, T, d_model) inputs with ``num_heads`` heads.

        ``mask=None`` uses the causal mask, matching ``infer()``. A custom mask
        is an additive bias broadcastable to (B, H, T, T); a query row that is
        entirely −∞ produces zero attention weights (see the module docstring).
        """
        B, T, C = x.shape          # C == d_model
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q(x)            # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (B, T, d_model) → (B, H, T, d_k)
        Q = ops.transpose(ops.reshape(Q, (B, T, H, d_k)), (0, 2, 1, 3))
        K = ops.transpose(ops.reshape(K, (B, T, H, d_k)), (0, 2, 1, 3))
        V = ops.transpose(ops.reshape(V, (B, T, H, d_k)), (0, 2, 1, 3))

        if self.rope is not None:
            Q = self.rope.rotate(Q)
            K = self.rope.rotate(K)

        # Scaled dot-product: (B, H, T, T)
        K_T = ops.transpose(K, (0, 1, 3, 2))             # (B, H, d_k, T)
        scores = ops.matmul(Q, K_T) * self.scale         # (B, H, T, T)

        if mask is None:
            mask = Tensor(_causal_mask(T, T, 0))
        else:
            mask = _prepare_mask(mask, scores.shape)
        scores = scores + mask  # (T, T) broadcasts over (B, H, T, T)

        weights = ops.softmax(scores)
        weights = self.attn_drop(weights)

        attn = ops.matmul(weights, V)                     # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) → (B, T, d_model)
        attn = ops.transpose(attn, (0, 2, 1, 3))          # (B, T, H, d_k)
        attn = ops.reshape(attn, (B, T, C))

        return self.out_proj(attn)

    def infer(self, x, cache=None, key_bias=None, positions=None):
        """
        NumPy-only inference with an optional key/value cache.

        ``key_bias`` is an additive per-key bias broadcastable to the
        (batch, heads, query, key) scores, used to hide padded keys.
        ``positions`` gives explicit RoPE positions for the new tokens (shape
        (batch, 1, query)); without it every row starts at the cache length.
        """
        B, T, C = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = self.W_k.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = self.W_v.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        past_len = 0 if cache is None else cache["k"].shape[2]
        if self.rope is not None:
            # Rotate new positions only — cached keys were rotated when added.
            if positions is None:
                Q = self.rope.rotate_np(Q, offset=past_len)
                K = self.rope.rotate_np(K, offset=past_len)
            else:
                Q = self.rope.rotate_np(Q, positions=positions)
                K = self.rope.rotate_np(K, positions=positions)
        if cache is not None:
            K = np.concatenate([cache["k"], K], axis=2)
            V = np.concatenate([cache["v"], V], axis=2)

        scores = Q @ K.transpose(0, 1, 3, 2) * self.scale
        scores += _causal_mask(T, K.shape[2], past_len)
        if key_bias is not None:
            scores = scores + key_bias
        weights = _softmax(scores)
        attn = weights @ V
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj.infer(attn), {"k": K, "v": V}


def _causal_mask(query_len, key_len, past_len):
    query_positions = past_len + np.arange(query_len)[:, None]
    key_positions = np.arange(key_len)[None, :]
    return np.where(key_positions > query_positions, -np.inf, 0.0)


def _prepare_mask(mask, scores_shape):
    """
    Validate an additive attention mask and return it as a Tensor.

    Accepts a Tensor or anything NumPy can turn into a float array. The mask
    must broadcast *up to* the score shape (it may be smaller, never larger),
    and may not contain NaN or +inf, which no masking scheme can mean and which
    would make the softmax NaN rather than simply dropping a key.
    """
    mask = mask if isinstance(mask, Tensor) else Tensor(mask)
    scores_shape = tuple(scores_shape)
    try:
        broadcast = np.broadcast_shapes(mask.shape, scores_shape)
    except ValueError as exc:
        raise ValueError(
            f"attention mask shape {mask.shape} does not broadcast to "
            f"attention scores {scores_shape}"
        ) from exc
    if broadcast != scores_shape:
        raise ValueError(
            f"attention mask shape {mask.shape} is larger than attention "
            f"scores {scores_shape}"
        )
    if not np.all(np.isfinite(mask.data) | np.isneginf(mask.data)):
        raise ValueError(
            "attention mask entries must be finite biases or -inf "
            "(NaN and +inf are not valid masks)"
        )
    return mask


def _softmax(x):
    """
    Row-wise softmax for the inference path.

    Mirrors ``ops.softmax``, including its definition of a fully masked row
    (every entry −∞) as all-zero weights, so ``forward()`` and ``infer()``
    cannot disagree about a masked position.
    """
    row_max = x.max(axis=-1, keepdims=True)
    shift = np.where(np.isneginf(row_max), 0.0, row_max)
    exp = np.exp(x - shift)
    total = exp.sum(axis=-1, keepdims=True)
    return exp / np.where(total == 0.0, 1.0, total)
