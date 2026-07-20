"""
layers.py — Linear, Embedding, LayerNorm, Dropout.
"""

import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from .module import Module


class Linear(Module):
    """
    Fully-connected layer: y = x @ Wᵀ + b

    Weight shape: (out_features, in_features) — matches PyTorch convention.
    Initialised with Kaiming uniform (fan-in scaling).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
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
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
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
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.02,
            requires_grad=True,
        )

    def forward(self, idx) -> Tensor:
        """idx: integer array of any shape → output shape (*idx.shape, embedding_dim)."""
        return self.weight[idx]

    def infer(self, idx):
        return self.weight.data[idx]

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
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mu = ops.mean(x, axis=-1, keepdims=True)        # (..., 1)
        diff = x - mu                                    # (..., C)
        var = ops.mean(diff ** 2, axis=-1, keepdims=True)  # (..., 1)
        x_hat = diff * ((var + self.eps) ** -0.5)       # (..., C)
        return x_hat * self.gamma + self.beta            # (..., C)

    def infer(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return (x - mu) * ((var + self.eps) ** -0.5) * self.gamma.data + self.beta.data

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class Dropout(Module):
    """
    Inverted dropout.  Acts as identity when p=0 or training=False.

    The binary mask is generated with NumPy (not tracked in the graph);
    the *multiplication* with the Tensor is tracked so gradients flow through.
    """

    def __init__(self, p: float = 0.0):
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
