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
"""

import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from .module import Module
from .layers import Linear, Dropout


class SelfAttention(Module):
    """Single-head causal self-attention."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        self.d_model = d_model
        self.scale = d_model ** -0.5

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.attn_drop = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: (B, T, d_model)
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # scores: (B, T, T)
        K_T = ops.transpose(K, (0, 2, 1))          # (B, d_model, T)
        scores = ops.matmul(Q, K_T) * self.scale    # (B, T, T)

        if mask is not None:
            scores = scores + mask  # (T, T) broadcasts to (B, T, T)

        weights = ops.softmax(scores)               # (B, T, T)
        weights = self.attn_drop(weights)

        out = ops.matmul(weights, V)                # (B, T, d_model)
        return self.out_proj(out)

    def infer(self, x, cache=None):
        """NumPy-only inference with an optional key/value cache."""
        Q = self.W_q.infer(x)
        K = self.W_k.infer(x)
        V = self.W_v.infer(x)
        past_len = 0 if cache is None else cache["k"].shape[1]
        if cache is not None:
            K = np.concatenate([cache["k"], K], axis=1)
            V = np.concatenate([cache["v"], V], axis=1)
        scores = Q @ np.swapaxes(K, -1, -2) * self.scale
        scores += _causal_mask(x.shape[1], K.shape[1], past_len)
        weights = _softmax(scores)
        return self.out_proj.infer(weights @ V), {"k": K, "v": V}


class MultiHeadAttention(Module):
    """
    Multi-head causal self-attention (GPT-style, pre-norm variant).

    Uses separate W_q, W_k, W_v projections for clarity, then merges heads.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = self.d_k ** -0.5

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.attn_drop = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        B, T, C = x.shape          # C == d_model
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q(x)            # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (B, T, d_model) → (B, H, T, d_k)
        Q = ops.transpose(ops.reshape(Q, (B, T, H, d_k)), (0, 2, 1, 3))
        K = ops.transpose(ops.reshape(K, (B, T, H, d_k)), (0, 2, 1, 3))
        V = ops.transpose(ops.reshape(V, (B, T, H, d_k)), (0, 2, 1, 3))

        # Scaled dot-product: (B, H, T, T)
        K_T = ops.transpose(K, (0, 1, 3, 2))             # (B, H, d_k, T)
        scores = ops.matmul(Q, K_T) * self.scale         # (B, H, T, T)

        if mask is not None:
            scores = scores + mask  # (T, T) broadcasts over (B, H, T, T)

        weights = ops.softmax(scores)
        weights = self.attn_drop(weights)

        attn = ops.matmul(weights, V)                     # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) → (B, T, d_model)
        attn = ops.transpose(attn, (0, 2, 1, 3))          # (B, T, H, d_k)
        attn = ops.reshape(attn, (B, T, C))

        return self.out_proj(attn)

    def infer(self, x, cache=None):
        """NumPy-only inference with an optional key/value cache."""
        B, T, C = x.shape
        H, d_k = self.num_heads, self.d_k

        Q = self.W_q.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = self.W_k.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = self.W_v.infer(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        past_len = 0 if cache is None else cache["k"].shape[2]
        if cache is not None:
            K = np.concatenate([cache["k"], K], axis=2)
            V = np.concatenate([cache["v"], V], axis=2)

        scores = Q @ K.transpose(0, 1, 3, 2) * self.scale
        scores += _causal_mask(T, K.shape[2], past_len)
        weights = _softmax(scores)
        attn = weights @ V
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj.infer(attn), {"k": K, "v": V}


def _causal_mask(query_len, key_len, past_len):
    query_positions = past_len + np.arange(query_len)[:, None]
    key_positions = np.arange(key_len)[None, :]
    return np.where(key_positions > query_positions, -1e9, 0.0)


def _softmax(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
