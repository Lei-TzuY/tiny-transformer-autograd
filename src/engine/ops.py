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
# div (element-wise)
# ---------------------------------------------------------------------------
def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division with NumPy broadcasting semantics."""
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        a.data / b.data,
        requires_grad=needs_grad,
        _children=(a, b),
        _op="div",
    )

    def _backward():
        if a.requires_grad:
            a._ensure_grad()
            a.grad += _unbroadcast(out.grad / b.data, a.shape)
        if b.requires_grad:
            b._ensure_grad()
            b.grad += _unbroadcast(
                -out.grad * a.data / (b.data * b.data), b.shape
            )

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiply with the same 1-D promotion and batch broadcasting as
    ``numpy.matmul``. Broadcast batch axes are summed away in the backward.
    """
    needs_grad = a.requires_grad or b.requires_grad
    out = Tensor(
        a.data @ b.data,
        requires_grad=needs_grad,
        _children=(a, b),
        _op="matmul",
    )

    def _backward():
        # NumPy temporarily treats a 1-D left operand as (1, K), and a 1-D
        # right operand as (K, 1), then squeezes the inserted output axes.
        # Recreate those promoted matrix shapes so the usual matrix-gradient
        # formula also covers vector-vector and vector-matrix products.
        a_was_vector = a.ndim == 1
        b_was_vector = b.ndim == 1
        a_matrix = a.data[np.newaxis, :] if a_was_vector else a.data
        b_matrix = b.data[..., np.newaxis] if b_was_vector else b.data

        grad_matrix = out.grad
        if a_was_vector and b_was_vector:
            grad_matrix = np.asarray(grad_matrix).reshape(1, 1)
        elif a_was_vector:
            grad_matrix = np.expand_dims(grad_matrix, axis=-2)
        elif b_was_vector:
            grad_matrix = np.expand_dims(grad_matrix, axis=-1)

        if a.requires_grad:
            a._ensure_grad()
            a_grad = np.matmul(grad_matrix, np.swapaxes(b_matrix, -1, -2))
            if a_was_vector:
                a_grad = np.squeeze(a_grad, axis=-2)
            a.grad += _unbroadcast(a_grad, a.shape)
        if b.requires_grad:
            b._ensure_grad()
            b_grad = np.matmul(np.swapaxes(a_matrix, -1, -2), grad_matrix)
            if b_was_vector:
                b_grad = np.squeeze(b_grad, axis=-1)
            b.grad += _unbroadcast(b_grad, b.shape)

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
    # np.where eagerly evaluates both branches, so directly spelling exp(-x)
    # and exp(x) still overflows in the unselected branch.  exp(-abs(x)) is
    # bounded by one and keeps every intermediate finite.
    z = np.exp(-np.abs(x.data))
    s = np.where(x.data >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
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
    """Natural logarithm on the positive real domain."""
    if np.isnan(x.data).any() or np.any(x.data <= 0.0):
        raise ValueError("log requires positive inputs")

    out = Tensor(
        np.log(x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="log",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad / x.data

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

    Fully masked rows
    -----------------
    A row of all −∞ has no probability distribution: ``max`` is −∞ and the
    textbook shift would evaluate ``-inf − -inf = nan``, poisoning the whole
    batch.  This softmax defines such a row as **all zeros** instead, the
    standard masked-attention convention: the query attends to nothing, so its
    context vector is zero and its gradient is zero (``S = 0`` makes the VJP
    above vanish).  Rows with at least one finite entry are unaffected.

    Invalid non-finite values
    -------------------------
    ``-inf`` is meaningful as an attention mask, but NaN and ``+inf`` are not:
    the latter would make the usual max-shift evaluate ``inf - inf``. Reject
    them explicitly instead of silently returning a NaN probability row.
    """
    if np.isnan(x.data).any() or np.isposinf(x.data).any():
        raise ValueError("softmax inputs must not contain NaN or +inf")

    row_max = x.data.max(axis=-1, keepdims=True)
    # Shift by 0 where no finite maximum exists so the exponent stays defined.
    shift = np.where(np.isneginf(row_max), 0.0, row_max)
    e = np.exp(x.data - shift)
    total = e.sum(axis=-1, keepdims=True)
    # A fully masked row sums to exactly 0; dividing by 1 keeps it all zeros
    # without emitting a divide-by-zero warning.
    s = e / np.where(total == 0.0, 1.0, total)
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
def cross_entropy(logits: Tensor, targets, ignore_index=None) -> Tensor:
    """
    Stable mean cross-entropy over the final class axis.

    Parameters
    ----------
    logits : Tensor, shape (..., C)
        Unnormalised class scores. The final axis contains the classes.
    targets : integer array-like, shape (...)
        One class index for every position in ``logits.shape[:-1]``.
    ignore_index : int or None
        Target value marking a position that must not contribute to the loss —
        typically the padding label of a variable-length batch. Ignored
        positions are excluded from the mean (the divisor is the number of
        *scored* positions) and receive exactly zero gradient, so padding can
        never influence training. ``ignore_index`` is the only target value
        allowed outside ``[0, C)``.

    Returns
    -------
    Tensor  scalar mean loss over the scored positions
    """
    if logits.ndim == 0:
        raise ValueError("cross_entropy logits must have a class dimension")
    if logits.data.size == 0 or logits.shape[-1] == 0:
        raise ValueError("cross_entropy inputs must be non-empty")

    targets_np = np.asarray(targets)
    expected_shape = logits.shape[:-1]
    if targets_np.shape != expected_shape:
        raise ValueError(
            "cross_entropy target shape mismatch: "
            f"expected {expected_shape}, got {targets_np.shape}"
        )
    if targets_np.size == 0:
        raise ValueError("cross_entropy targets must be non-empty")
    if not np.issubdtype(targets_np.dtype, np.integer):
        raise TypeError("cross_entropy targets must contain integers")
    if ignore_index is not None and not isinstance(ignore_index, (int, np.integer)):
        raise TypeError("cross_entropy ignore_index must be an integer or None")

    num_classes = logits.shape[-1]
    # Backward must use the labels seen by this forward pass even if a caller
    # later mutates the original array in place.
    targets_flat = np.array(targets_np, dtype=np.int64, copy=True).reshape(-1)

    # Select the positions that are actually scored. ``rows is None`` marks the
    # common "everything is scored" case so no row gathering happens at all.
    rows = None
    if ignore_index is not None:
        scored = targets_flat != int(ignore_index)
        if not scored.any():
            raise ValueError(
                "cross_entropy has no scored target: every position equals "
                f"ignore_index={ignore_index}"
            )
        if not scored.all():
            rows = np.flatnonzero(scored)
            targets_flat = targets_flat[rows]

    if np.any(targets_flat < 0) or np.any(targets_flat >= num_classes):
        raise ValueError(
            f"cross_entropy targets must be in [0, {num_classes})"
        )

    # Flatten only the leading sample axes. Subtracting the row maximum
    # implements log-sum-exp without exponentiating a large positive value;
    # unlike log(softmax + eps), this forward exactly matches the standard
    # softmax-minus-one-hot derivative for extremely unlikely targets.
    logits_flat = logits.data.reshape(-1, num_classes)
    scored_logits = logits_flat if rows is None else logits_flat[rows]
    if np.isnan(scored_logits).any() or np.isposinf(scored_logits).any():
        raise ValueError(
            "cross_entropy scored logits must not contain NaN or +inf"
        )

    row_max = scored_logits.max(axis=-1, keepdims=True)
    # softmax() defines an all −∞ row as zero attention weights, but a loss has
    # no such reading: every class is impossible, so no target can be scored.
    # Fail loudly instead of returning a NaN that would spread to every weight.
    # Only scored rows matter — an ignored position may be anything.
    if np.isneginf(row_max).any():
        raise ValueError(
            "cross_entropy requires at least one finite logit per scored row"
        )
    shifted = scored_logits - row_max
    exponentials = np.exp(shifted)
    normalisers = exponentials.sum(axis=-1, keepdims=True)
    probabilities = exponentials / normalisers
    sample_count = targets_flat.size
    target_shifted = shifted[np.arange(sample_count), targets_flat]
    loss_val = (np.log(normalisers[:, 0]) - target_shifted).mean()

    out = Tensor(
        loss_val,
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="cross_entropy",
    )

    def _backward():
        if logits.requires_grad:
            logits._ensure_grad()
            # ∂L/∂logits = (softmax − one_hot) / scored_count
            scored_grad = probabilities.copy()
            scored_grad[np.arange(sample_count), targets_flat] -= 1.0
            scored_grad /= sample_count
            if rows is None:
                grad_logits = scored_grad
            else:
                # Ignored positions keep an exactly zero gradient.
                grad_logits = np.zeros_like(logits_flat)
                grad_logits[rows] = scored_grad
            logits.grad += out.grad * grad_logits.reshape(logits.shape)

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
    axes_tuple = None if axes is None else tuple(axes)
    out = Tensor(
        np.transpose(x.data, axes_tuple),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="transpose",
    )

    if axes_tuple is None:
        inverse_axes = None
    else:
        # np.transpose above performs length, uniqueness, and bounds checks.
        # Normalise valid negative axes before computing the inverse
        # permutation; argsort on raw negatives is not an inverse.
        normalised_axes = tuple(axis % x.ndim for axis in axes_tuple)
        inverse_axes = tuple(np.argsort(normalised_axes))

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += np.transpose(out.grad, inverse_axes)

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
# silu / swish  (Elfwing et al. 2017; used in SwiGLU feed-forwards)
# ---------------------------------------------------------------------------
def silu(x: Tensor) -> Tensor:
    """
    SiLU(x) = x · σ(x)
    Backward: d/dx = σ(x) + x·σ(x)·(1−σ(x)) = σ(x)·(1 + x·(1−σ(x)))
    Uses the same numerically stable sigmoid as sigmoid().
    """
    s = np.where(
        x.data >= 0,
        1.0 / (1.0 + np.exp(-np.abs(x.data))),
        np.exp(-np.abs(x.data)) / (1.0 + np.exp(-np.abs(x.data))),
    )
    out = Tensor(
        x.data * s, requires_grad=x.requires_grad, _children=(x,), _op="silu"
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            x.grad += out.grad * (s + x.data * s * (1.0 - s))

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
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("concat requires at least one tensor")
    needs_grad = any(t.requires_grad for t in tensors)
    out = Tensor(
        np.concatenate([t.data for t in tensors], axis=axis),
        requires_grad=needs_grad,
        _children=tensors,
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