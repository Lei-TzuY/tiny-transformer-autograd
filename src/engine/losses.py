"""Numerically stable log-probability and classification loss primitives."""

import numpy as np

from .ops import _stable_sum_data
from .tensor import Tensor


_VALID_REDUCTIONS = {"none", "sum", "mean"}


def _normalise_axis(axis, ndim):
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("log_softmax axis must be an integer")
    if ndim == 0:
        raise ValueError("log_softmax input must have at least one dimension")
    axis = int(axis)
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"log_softmax axis {axis} is out of bounds for tensor with {ndim} dimensions"
        )
    return axis % ndim


def _validate_reduction(reduction):
    if not isinstance(reduction, str):
        raise TypeError("loss reduction must be a string")
    if reduction not in _VALID_REDUCTIONS:
        choices = ", ".join(sorted(_VALID_REDUCTIONS))
        raise ValueError(f"loss reduction must be one of: {choices}")
    return reduction


def _validate_ignore_index(ignore_index):
    if ignore_index is None:
        return None
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("loss ignore_index must be an integer or None")
    return int(ignore_index)


def _validate_smoothing(smoothing):
    if isinstance(smoothing, (bool, np.bool_)) or not isinstance(
        smoothing, (int, float, np.integer, np.floating)
    ):
        raise TypeError("label smoothing must be a real number")
    smoothing = float(smoothing)
    if not np.isfinite(smoothing):
        raise ValueError("label smoothing must be finite")
    if smoothing < 0.0 or smoothing > 1.0:
        raise ValueError("label smoothing must be in [0, 1]")
    return smoothing


def _stable_mean_data(values, axis=None, keepdims=False):
    """Mean finite values without a recoverable sum-before-divide overflow."""
    values = np.asarray(values, dtype=np.float64)
    if axis is None:
        count = values.size
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = int(np.prod([values.shape[item] for item in axes], dtype=np.int64))
    if count == 0:
        raise ValueError("loss mean reduction has no elements")

    with np.errstate(over="ignore", invalid="ignore"):
        historical = np.mean(values, axis=axis, keepdims=keepdims)
    if np.isfinite(historical).all() or not np.isfinite(values).all():
        return historical

    # Scaling before the sum makes cases such as mean([1e308, 1e308])
    # representable even though the historical intermediate sum is +inf.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = values * (1.0 / count)
    fallback = _stable_sum_data(scaled, axis=axis, keepdims=keepdims)
    return np.where(np.isfinite(fallback), fallback, historical)


def _prepare_nll_inputs(log_probs, targets, ignore_index):
    if not isinstance(log_probs, Tensor):
        raise TypeError("nll_loss log_probs must be a Tensor")
    if log_probs.ndim == 0:
        raise ValueError("nll_loss log_probs must have a class dimension")
    if log_probs.data.size == 0 or log_probs.shape[-1] == 0:
        raise ValueError("nll_loss inputs must be non-empty")

    targets_np = np.asarray(targets)
    expected_shape = log_probs.shape[:-1]
    if targets_np.shape != expected_shape:
        raise ValueError(
            "nll_loss target shape mismatch: "
            f"expected {expected_shape}, got {targets_np.shape}"
        )
    if targets_np.size == 0:
        raise ValueError("nll_loss targets must be non-empty")
    if not np.issubdtype(targets_np.dtype, np.integer):
        raise TypeError("nll_loss targets must contain integers")

    ignore_index = _validate_ignore_index(ignore_index)
    num_classes = log_probs.shape[-1]
    targets_flat = np.array(targets_np, dtype=np.int64, copy=True).reshape(-1)
    total_positions = targets_flat.size

    rows = None
    if ignore_index is not None:
        scored = targets_flat != ignore_index
        if not scored.any():
            raise ValueError(
                "nll_loss has no scored target: every position equals "
                f"ignore_index={ignore_index}"
            )
        if not scored.all():
            rows = np.flatnonzero(scored)
            targets_flat = targets_flat[rows]

    if np.any(targets_flat < 0) or np.any(targets_flat >= num_classes):
        raise ValueError(f"nll_loss targets must be in [0, {num_classes})")

    log_probs_flat = log_probs.data.reshape(-1, num_classes)
    scored_log_probs = log_probs_flat if rows is None else log_probs_flat[rows]
    # -inf is a valid log-probability for an impossible class. NaN and +inf
    # have no probability interpretation and would make losses ambiguous.
    if np.isnan(scored_log_probs).any() or np.isposinf(scored_log_probs).any():
        raise ValueError("nll_loss scored log_probs must not contain NaN or +inf")

    active_rows = (
        np.arange(total_positions, dtype=np.int64) if rows is None else rows
    )
    return (
        targets_flat,
        rows,
        active_rows,
        scored_log_probs,
        num_classes,
        total_positions,
    )


def _reduce_scored_losses(scored_losses, rows, total_positions, output_shape, reduction):
    if reduction == "none":
        losses = np.zeros(total_positions, dtype=np.float64)
        if rows is None:
            losses[:] = scored_losses
        else:
            losses[rows] = scored_losses
        return losses.reshape(output_shape)
    if reduction == "sum":
        return _stable_sum_data(scored_losses)
    return _stable_mean_data(scored_losses)


def _loss_row_multipliers(out, rows, total_positions, scored_count, reduction):
    if reduction == "none":
        upstream = np.asarray(out.grad, dtype=np.float64).reshape(-1)
        return upstream if rows is None else upstream[rows]

    scalar = float(np.asarray(out.grad))
    multipliers = np.full(scored_count, scalar, dtype=np.float64)
    if reduction == "mean":
        multipliers /= scored_count
    return multipliers


def log_softmax(x: Tensor, axis=-1) -> Tensor:
    """Stable log-softmax with a closed-form reverse-mode VJP.

    ``-inf`` entries are accepted as masked/impossible classes when another
    finite value exists in the same normalization slice. NaN, ``+inf``, and an
    all-``-inf`` slice are rejected because they do not define a probability
    distribution.
    """
    if not isinstance(x, Tensor):
        raise TypeError("log_softmax input must be a Tensor")
    axis = _normalise_axis(axis, x.ndim)
    if x.shape[axis] == 0:
        raise ValueError("log_softmax normalization axis must be non-empty")
    if np.isnan(x.data).any() or np.isposinf(x.data).any():
        raise ValueError("log_softmax inputs must not contain NaN or +inf")

    row_max = np.max(x.data, axis=axis, keepdims=True)
    if np.isneginf(row_max).any():
        raise ValueError(
            "log_softmax requires at least one finite value per normalization slice"
        )

    # Subtracting opposite-sign finite extremes can itself overflow to -inf.
    # That result is semantically correct (the probability rounds to zero), so
    # suppress only the intermediate warning rather than clipping the logits.
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = np.asarray(x.data) - row_max
    exponentials = np.exp(shifted)
    normalisers = _stable_sum_data(exponentials, axis=axis, keepdims=True)
    log_normalisers = np.log(normalisers)
    values = shifted - log_normalisers
    probabilities = exponentials / normalisers

    out = Tensor(
        values,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="log_softmax",
    )

    def _backward():
        if x.requires_grad:
            x._ensure_grad()
            grad_sum = _stable_sum_data(out.grad, axis=axis, keepdims=True)
            x.grad += out.grad - probabilities * grad_sum

    out._backward = _backward
    return out


def nll_loss(log_probs: Tensor, targets, ignore_index=None, reduction="mean") -> Tensor:
    """Negative log-likelihood over the final class axis.

    ``reduction`` may be ``"none"``, ``"sum"``, or ``"mean"``. Ignored
    positions are zero for ``"none"`` and excluded from scalar reductions.
    """
    reduction = _validate_reduction(reduction)
    (
        targets_flat,
        rows,
        active_rows,
        scored_log_probs,
        num_classes,
        total_positions,
    ) = _prepare_nll_inputs(log_probs, targets, ignore_index)

    scored_count = targets_flat.size
    selected = scored_log_probs[np.arange(scored_count), targets_flat]
    scored_losses = -selected
    loss_value = _reduce_scored_losses(
        scored_losses, rows, total_positions, log_probs.shape[:-1], reduction
    )

    out = Tensor(
        loss_value,
        requires_grad=log_probs.requires_grad,
        _children=(log_probs,),
        _op="nll_loss",
    )

    def _backward():
        if log_probs.requires_grad:
            log_probs._ensure_grad()
            multipliers = _loss_row_multipliers(
                out, rows, total_positions, scored_count, reduction
            )
            grad_flat = np.zeros((total_positions, num_classes), dtype=np.float64)
            grad_flat[active_rows, targets_flat] = -multipliers
            log_probs.grad += grad_flat.reshape(log_probs.shape)

    out._backward = _backward
    return out


def _label_smoothed_nll_loss(
    log_probs: Tensor,
    targets,
    *,
    smoothing,
    ignore_index=None,
    reduction="mean",
) -> Tensor:
    reduction = _validate_reduction(reduction)
    (
        targets_flat,
        rows,
        active_rows,
        scored_log_probs,
        num_classes,
        total_positions,
    ) = _prepare_nll_inputs(log_probs, targets, ignore_index)

    scored_count = targets_flat.size
    target_losses = -scored_log_probs[np.arange(scored_count), targets_flat]
    smooth_losses = _stable_mean_data(-scored_log_probs, axis=1)
    if smoothing == 1.0:
        scored_losses = smooth_losses
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            scored_losses = (
                (1.0 - smoothing) * target_losses
                + smoothing * smooth_losses
            )

    loss_value = _reduce_scored_losses(
        scored_losses, rows, total_positions, log_probs.shape[:-1], reduction
    )
    out = Tensor(
        loss_value,
        requires_grad=log_probs.requires_grad,
        _children=(log_probs,),
        _op="label_smoothed_nll_loss",
    )

    def _backward():
        if log_probs.requires_grad:
            log_probs._ensure_grad()
            multipliers = _loss_row_multipliers(
                out, rows, total_positions, scored_count, reduction
            )
            scored_grad = np.full(
                (scored_count, num_classes),
                -smoothing / num_classes,
                dtype=np.float64,
            )
            scored_grad[np.arange(scored_count), targets_flat] -= 1.0 - smoothing
            scored_grad *= multipliers[:, None]

            grad_flat = np.zeros((total_positions, num_classes), dtype=np.float64)
            grad_flat[active_rows] = scored_grad
            log_probs.grad += grad_flat.reshape(log_probs.shape)

    out._backward = _backward
    return out


def label_smoothed_cross_entropy(
    logits: Tensor,
    targets,
    smoothing=0.1,
    ignore_index=None,
    reduction="mean",
) -> Tensor:
    """Cross-entropy against a one-hot/uniform label mixture.

    The target distribution is ``(1-smoothing) * one_hot + smoothing / C``.
    ``smoothing=0`` is therefore ordinary NLL over :func:`log_softmax`, while
    ``smoothing=1`` trains against the uniform distribution.
    """
    smoothing = _validate_smoothing(smoothing)
    reduction = _validate_reduction(reduction)
    log_probs = log_softmax(logits, axis=-1)
    if smoothing == 0.0:
        return nll_loss(
            log_probs,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )
    return _label_smoothed_nll_loss(
        log_probs,
        targets,
        smoothing=smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
    )
