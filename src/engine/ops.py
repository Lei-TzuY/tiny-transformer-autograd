"""
ops.py — Primitive differentiable operations.

Each function takes Tensor inputs, returns a Tensor output, and registers
a _backward closure that accumulates gradients into the parents' .grad
fields.

Broadcasting note
-----------------
NumPy broadcasts automatically in the forward pass.  In the backward pass
we must *un-broadcast* (sum over the axes that were broadcast) before
accumulating into the parent's .grad which has a smaller shape.

Gradient formulae
-----------------
add(a, b)      : ∂L/∂a = grad, ∂L/∂b = grad
mul(a, b)      : ∂L/∂a = grad * b, ∂L/∂b = grad * a
matmul(a, b)   : ∂L/∂a = grad @ bᵀ, ∂L/∂b = aᵀ @ grad
relu(x)        : ∂L/∂x = grad * (x > 0)
exp(x)         : ∂L/∂x = grad * exp(x)
log(x)         : ∂L/∂x = grad / x
softmax(x)     : ∂L/∂xᵢ = Sᵢ(δᵢⱼ - Sⱼ) · gradⱼ  →  S*(grad - (grad·S).sum())
cross_entropy  : combined softmax + NLL; ∂L/∂logits = (softmax - one_hot) / N
"""

import numpy as np
from .tensor import Tensor


# ---------------------------------------------------------------------------
# Helper: un-broadcast a gradient to match a target shape
# ---------------------------------------------------------------------------
def _unbroadcast(grad, target_shape):
    """Sum gradient axes that were broadcast to match target_shape."""
    # Pad target_shape on the left with 1s to match grad ndim
    ndim_diff = grad.ndim - len(target_shape)
    padded = (1,) * ndim_diff + tuple(target_shape)

    # Sum over axes where target_shape had size 1 (or was added)
    axes = tuple(i for i, (g, t) in enumerate(zip(grad.shape, padded)) if t == 1)
    result = grad.sum(axis=axes, keepdims=True) if axes else grad
    return result.reshape(target_shape)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
def add(a: Tensor, b: Tensor) -> Tensor:
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        a.data + b.data,
        requires_grad=needs_grad,
        _children=(a, b),
        _op="add",
    )

    def _backward():
        if a.requires_grad:
            a._ensure_grad()
            a.grad += _unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b._ensure_grad()
            b.grad += _unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# mul (element-wise)
# ---------------------------------------------------------------------------
def mul(a: Tensor, b: Tensor) -> Tensor:
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        a.data * b.data,
        requires_grad=needs_grad,
        _children=(a, b),
        _op="mul",
    )

    def _backward():
        if a.requires_grad:
            a._ensure_grad()
            a.grad += _unbroadcast(out.grad * b.data, a.shape)
        if b.requires_grad:
            b._ensure_grad()
            b.grad += _unbroadcast(out.grad * a.data, b.shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiply supporting 2-D and batched tensors.
    For a: (..., M, K) @ b: (..., K, N) → (..., M, N)
    ∂L/∂a = grad @ bᵀ   →  (..., M, K)
    ∂L/∂b = aᵀ @ grad   →  (..., K, N)
    Batch dims broadcast: extra leading dims are summed away in the backward.
    """
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        a.data @ b.data,
        requires_grad=needs_grad,
        _children=(a, b),
        _op="matmul",
    )

    def _backward():
        if a.requires_grad:
            a._ensure_grad()
            a_grad = out.grad @ np.swapaxes(b.data, -1, -2)
            # Sum away any extra batch dims that were broadcast over
            while a_grad.ndim > a.data.ndim:
                a_grad = a_grad.sum(axis=0)
            a.grad += a_grad
        if b.requires_grad:
            b._ensure_grad()
            b_grad = np.swapaxes(a.data, -1, -2) @ out.grad
            while b_grad.ndim > b.data.ndim:
                b_grad = b_grad.sum(axis=0)
            b.grad += b_grad

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------
def sigmoid(x: Tensor) -> Tensor:
    """
    σ(x) = 1 / (1 + e^{−x})
    Numerically stable: uses e^x / (1 + e^x) for negative inputs to avoid
    overflow in e^{−x}.
    Backward: dL/dx = grad · σ(x) · (1 − σ(x))
    """
    s = np.where(
        x.data >= 0,
        1.0 / (1.0 + np.exp(-x.data)),
        np.exp(x.data) / (1.0 + np.exp(x.data)),
    )
    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="sigmoid")

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------
def relu(x: Tensor) -> Tensor:
    out = Tensor(
        np.maximum(0.0, x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="relu",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad * (x.data > 0)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# exp
# ---------------------------------------------------------------------------
def exp(x: Tensor) -> Tensor:
    e = np.exp(x.data)
    out = Tensor(e, requires_grad=x.requires_grad, _children=(x,), _op="exp")

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad * e  # d/dx exp(x) = exp(x)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# log  (natural log)
# ---------------------------------------------------------------------------
def log(x: Tensor) -> Tensor:
    out = Tensor(
        np.log(x.data + 1e-12),  # numerical stability
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="log",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad / (x.data + 1e-12)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# softmax  (along last axis)
# ---------------------------------------------------------------------------
def softmax(x: Tensor) -> Tensor:
    """
    S = exp(x - max(x)) / sum(exp(x - max(x)))    (numerically stable)
    Backward:
      ∂L/∂xᵢ = Sᵢ · (∂L/∂Sᵢ  −  Σⱼ ∂L/∂Sⱼ · Sⱼ)
             = S ⊙ (grad − (grad · S).sum(axis=-1, keepdims=True))
    """
    shifted = x.data - x.data.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            dot = (out.grad * s).sum(axis=-1, keepdims=True)
            x.grad += s * (out.grad - dot)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# cross_entropy  (logits → scalar mean NLL loss, integer class targets)
# ---------------------------------------------------------------------------
def cross_entropy(logits: Tensor, targets) -> Tensor:
    """
    Combined softmax + negative log-likelihood for numerical stability.

    Parameters
    ----------
    logits : Tensor  shape (N, C)
    targets : array-like of int  shape (N,)   class indices

    Returns
    -------
    Tensor  scalar mean loss
    """
    targets_np = np.array(targets, dtype=np.int64)
    N = logits.data.shape[0]

    # Stable softmax
    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=-1, keepdims=True)  # (N, C)

    # NLL: -log(s[i, targets[i]])
    log_probs = np.log(s[np.arange(N), targets_np] + 1e-12)
    loss_val = -log_probs.mean()

    out = Tensor(
        loss_val,
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="cross_entropy",
    )

    def _backward():
        if logits.requires_grad:
            logits._ensure_grad()
            # ∂L/∂logits = (softmax − one_hot) / N
            grad_logits = s.copy()
            grad_logits[np.arange(N), targets_np] -= 1.0
            grad_logits /= N
            logits.grad += out.grad * grad_logits

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------
def sum(x: Tensor, axis=None, keepdims=False) -> Tensor:
    out = Tensor(
        x.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="sum",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            x.grad += np.broadcast_to(grad, x.shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------
def mean(x: Tensor, axis=None, keepdims=False) -> Tensor:
    if axis is None:
        n = x.data.size
    elif isinstance(axis, int):
        n = x.data.shape[axis]
    else:
        n = np.prod([x.data.shape[a] for a in axis])

    out_sum = sum(x, axis=axis, keepdims=keepdims)
    # Reuse mul with scalar  (1/n is a constant, no graph node needed)
    scalar = Tensor(np.array(1.0 / n))
    result = mul(out_sum, scalar)
    result._op = "mean"
    return result


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------
def reshape(x: Tensor, new_shape) -> Tensor:
    out = Tensor(
        x.data.reshape(new_shape),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="reshape",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad.reshape(x.shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# transpose (swap two axes)
# ---------------------------------------------------------------------------
def transpose(x: Tensor, axes=None) -> Tensor:
    out = Tensor(
        np.transpose(x.data, axes),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="transpose",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            if axes is None:
                inv_axes = None
            else:
                inv_axes = np.argsort(axes)
            x.grad += np.transpose(out.grad, inv_axes)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# tanh
# ---------------------------------------------------------------------------
def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _op="tanh")

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad * (1.0 - t * t)  # sech²(x)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# gelu  (tanh approximation — Hendrycks & Gimpel 2016)
# ---------------------------------------------------------------------------
def gelu(x: Tensor) -> Tensor:
    """GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))"""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x.data + 0.044715 * x.data ** 3)
    t = np.tanh(inner)
    val = 0.5 * x.data * (1.0 + t)
    out = Tensor(val, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            sech2 = 1.0 - t * t
            dtanh_dx = c * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            dx = 0.5 * (1.0 + t) + 0.5 * x.data * sech2 * dtanh_dx
            x.grad += out.grad * dx

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# concat along axis
# ---------------------------------------------------------------------------
def concat(tensors, axis=0) -> Tensor:
    needs_grad = any(t.requires_grad for t in tensors)
    out = Tensor(
        np.concatenate([t.data for t in tensors], axis=axis),
        requires_grad=needs_grad,
        _children=tuple(tensors),
        _op="concat",
    )

    # Precompute split sizes
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        parts = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)
        for t, p in zip(tensors, parts):
            if t.requires_grad:
                t._ensure_grad()
                t.grad += p

    out._backward = _backward
    return out
