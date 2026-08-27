"""
layers.py — Linear, Embedding, LayerNorm, RMSNorm, Dropout.
"""

from numbers import Integral, Real

import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from .module import Module


_NORMALIZATION_SCALE_THRESHOLD = np.sqrt(np.finfo(np.float64).max) / 2.0


def _normalization_scale(data):
    """Return per-row scaling only when centering/squaring can overflow."""
    max_abs = np.max(np.abs(data), axis=-1, keepdims=True)
    needs_scaling = np.isfinite(max_abs) & (
        max_abs > _NORMALIZATION_SCALE_THRESHOLD
    )
    if not np.any(needs_scaling):
        return None
    return np.where(needs_scaling, max_abs, 1.0)


def _constant_finite_rows(data):
    """Identify finite rows whose centered LayerNorm activation is exactly zero."""
    return np.all(np.isfinite(data), axis=-1, keepdims=True) & np.all(
        data == data[..., :1], axis=-1, keepdims=True
    )


def _layernorm_scale(data):
    """Keep exact constant rows in original units so epsilon cannot underflow."""
    scale = _normalization_scale(data)
    if scale is None:
        return None
    scale = np.where(_constant_finite_rows(data), 1.0, scale)
    return None if np.all(scale == 1.0) else scale


def _positive_int(name, value):
    """Validate a positive integral layer dimension and normalize it to int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bool_flag(name, value):
    """Validate a boolean layer option and normalize NumPy booleans."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _real_scalar(
    name,
    value,
    *,
    positive=False,
    lower=None,
    upper=None,
    upper_inclusive=True,
):
    """Validate one finite real layer hyperparameter and return float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None:
        invalid = value > upper if upper_inclusive else value >= upper
        if invalid:
            relation = "at most" if upper_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {upper}")
    return value


class Linear(Module):
    """
    Fully-connected layer: y = x @ Wᵀ + b

    Weight shape: (out_features, in_features) — matches PyTorch convention.
    Initialised with Kaiming uniform (fan-in scaling).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        in_features = _positive_int("in_features", in_features)
        out_features = _positive_int("out_features", out_features)
        bias = _bool_flag("bias", bias)
        self.in_features = in_features
        self.out_features = out_features

        k = np.sqrt(1.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_features, in_features)),
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        )
        self.lora_A = None
        self.lora_B = None
        self.lora_scaling = 1.0

    def forward(self, x: Tensor) -> Tensor:
        # ops.transpose keeps weight in the graph so its gradient is computed
        w_t = ops.transpose(self.weight, (1, 0))   # (in, out)
        out = ops.matmul(x, w_t)                   # (..., out)
        if self.bias is not None:
            out = out + self.bias
        if self.lora_A is not None:
            a_t = ops.transpose(self.lora_A, (1, 0))
            b_t = ops.transpose(self.lora_B, (1, 0))
            out = out + ops.matmul(ops.matmul(x, a_t), b_t) * self.lora_scaling
        return out

    def infer(self, x):
        """NumPy-only forward pass for generation."""
        out = x @ self.weight.data.T
        if self.bias is not None:
            out = out + self.bias.data
        if self.lora_A is not None:
            out = out + (x @ self.lora_A.data.T @ self.lora_B.data.T) * self.lora_scaling
        return out

    def enable_lora(self, rank, alpha=1.0):
        """Freeze this linear layer and add trainable low-rank adapters."""
        rank = _positive_int("LoRA rank", rank)
        alpha = _real_scalar("LoRA alpha", alpha, positive=True)
        if self.lora_A is not None:
            return
        self.weight.requires_grad = False
        self.weight.grad = None
        if self.bias is not None:
            self.bias.requires_grad = False
            self.bias.grad = None
        self.lora_A = Tensor(
            np.random.randn(rank, self.in_features) * 0.02,
            requires_grad=True,
        )
        self.lora_B = Tensor(
            np.zeros((self.out_features, rank)),
            requires_grad=True,
        )
        self.lora_scaling = alpha / rank

    def __repr__(self):
        return (
            f"Linear(in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None})"
        )


class Embedding(Module):
    """
    Lookup-table embedding: integer indices → dense row vectors.

    Gradient flows back via np.add.at (handles repeated indices correctly).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        num_embeddings = _positive_int("num_embeddings", num_embeddings)
        embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.02,
            requires_grad=True,
        )

    def forward(self, idx) -> Tensor:
        """idx: integer array of any shape → output shape (*idx.shape, embedding_dim)."""
        idx = self._validate_indices(idx)
        return self.weight[idx]

    def infer(self, idx):
        idx = self._validate_indices(idx)
        return self.weight.data[idx]

    def _validate_indices(self, idx):
        idx = np.asarray(idx)
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError("embedding indices must be integers")
        if np.any(idx < 0) or np.any(idx >= self.num_embeddings):
            raise ValueError(
                f"embedding indices must be in [0, {self.num_embeddings})"
            )
        return idx

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class LayerNorm(Module):
    """
    Layer normalisation over the last axis.

    y = (x − μ) / √(σ² + ε) · γ + β

    Gradients flow through all operations using the existing autograd primitives,
    so no custom backward is needed here.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        normalized_shape = _positive_int("normalized_shape", normalized_shape)
        eps = _real_scalar("eps", eps, positive=True)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input_shape(x.shape)
        scale = _layernorm_scale(x.data)
        if scale is None:
            mu = ops.mean(x, axis=-1, keepdims=True)        # (..., 1)
            diff = x - mu                                    # (..., C)
            var = ops.mean(diff ** 2, axis=-1, keepdims=True)  # (..., 1)
            x_hat = diff * ((var + self.eps) ** -0.5)       # (..., C)
            return x_hat * self.gamma + self.beta            # (..., C)

        scaled = x / scale
        mu = ops.mean(scaled, axis=-1, keepdims=True)
        diff = scaled - mu
        var = ops.mean(diff ** 2, axis=-1, keepdims=True)
        eps_scaled = (self.eps / scale) / scale
        x_hat = diff * ((var + eps_scaled) ** -0.5)
        return x_hat * self.gamma + self.beta

    def infer(self, x):
        x = np.asarray(x)
        self._validate_input_shape(x.shape)
        scale = _normalization_scale(x)
        if scale is None:
            mu = x.mean(axis=-1, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
            return (x - mu) * ((var + self.eps) ** -0.5) * self.gamma.data + self.beta.data

        # Exact constant finite rows normalize to zero regardless of magnitude.
        # Evaluate them through a zero surrogate so NumPy never has to sum huge
        # equal values or divide epsilon by a scale large enough to underflow it.
        constant_rows = _constant_finite_rows(x)
        safe_scale = np.where(constant_rows, 1.0, scale)
        scaled = np.where(constant_rows, 0.0, x / safe_scale)
        mu = scaled.mean(axis=-1, keepdims=True)
        diff = scaled - mu
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        eps_scaled = (self.eps / safe_scale) / safe_scale
        return diff * ((var + eps_scaled) ** -0.5) * self.gamma.data + self.beta.data

    def _validate_input_shape(self, shape):
        if not shape or shape[-1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected final dimension {self.normalized_shape}, "
                f"got shape {shape}"
            )

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class RMSNorm(Module):
    """
    Root-mean-square normalisation (Zhang & Sennrich 2019), used by Llama.

    y = x / √(mean(x²) + ε) · γ

    Unlike LayerNorm there is no mean-centering and no bias — cheaper, and in
    practice trains just as well.  Gradients flow through existing autograd
    primitives, so no custom backward is needed.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        normalized_shape = _positive_int("normalized_shape", normalized_shape)
        eps = _real_scalar("eps", eps, positive=True)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input_shape(x.shape)
        scale = _normalization_scale(x.data)
        if scale is None:
            ms = ops.mean(x ** 2, axis=-1, keepdims=True)   # (..., 1)
            return x * ((ms + self.eps) ** -0.5) * self.gamma

        scaled = x / scale
        ms = ops.mean(scaled ** 2, axis=-1, keepdims=True)
        eps_scaled = (self.eps / scale) / scale
        return scaled * ((ms + eps_scaled) ** -0.5) * self.gamma

    def infer(self, x):
        x = np.asarray(x)
        self._validate_input_shape(x.shape)
        scale = _normalization_scale(x)
        if scale is None:
            ms = (x ** 2).mean(axis=-1, keepdims=True)
            return x * ((ms + self.eps) ** -0.5) * self.gamma.data

        scaled = x / scale
        ms = (scaled ** 2).mean(axis=-1, keepdims=True)
        eps_scaled = (self.eps / scale) / scale
        return scaled * ((ms + eps_scaled) ** -0.5) * self.gamma.data

    def _validate_input_shape(self, shape):
        if not shape or shape[-1] != self.normalized_shape:
            raise ValueError(
                f"RMSNorm expected final dimension {self.normalized_shape}, "
                f"got shape {shape}"
            )

    def __repr__(self):
        return f"RMSNorm({self.normalized_shape}, eps={self.eps})"


class Dropout(Module):
    """
    Inverted dropout.  Acts as identity when p=0 or training=False.

    The binary mask is generated with NumPy (not tracked in the graph);
    the *multiplication* with the Tensor is tracked so gradients flow through.
    """

    def __init__(self, p: float = 0.0):
        p = _real_scalar(
            "dropout probability", p, lower=0.0, upper=1.0, upper_inclusive=False
        )
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep).astype(np.float64) / keep
        return x * Tensor(mask)

    def infer(self, x):
        return x

    def __repr__(self):
        return f"Dropout(p={self.p})"