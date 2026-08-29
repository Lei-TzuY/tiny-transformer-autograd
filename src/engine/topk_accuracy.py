"""Deterministic, read-only top-k accuracy for NumPy classification logits."""

from numbers import Integral

import numpy as np


def _strict_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_logits(logits):
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a NumPy array")
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if logits.shape[-1] == 0:
        raise ValueError("logits must have at least one class")
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("logits must have a floating dtype")


def _validate_targets(targets, expected_shape):
    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a NumPy array")
    if targets.shape != expected_shape:
        raise ValueError(
            f"targets shape {targets.shape} must match logits batch shape {expected_shape}"
        )
    if not np.issubdtype(targets.dtype, np.integer) or np.issubdtype(
        targets.dtype, np.bool_
    ):
        raise TypeError("targets must have an integer dtype")


def topk_accuracy_report(logits, targets, *, k=1, ignore_index=None):
    """Return a strict-JSON-safe top-k accuracy report.

    The final logits axis is the class axis. Ties are resolved deterministically by
    smaller class index, matching stable descending sorting of the original class
    order. Ignored target positions are excluded before logit finiteness checks, so
    padded rows may contain arbitrary floating values without affecting the metric.
    """
    _validate_logits(logits)
    _validate_targets(targets, logits.shape[:-1])

    k = _strict_integer("k", k)
    class_count = logits.shape[-1]
    if not 1 <= k <= class_count:
        raise ValueError("k must be between 1 and the number of classes")

    if ignore_index is not None:
        ignore_index = _strict_integer("ignore_index", ignore_index)
        scored_mask = targets != ignore_index
    else:
        scored_mask = np.ones(targets.shape, dtype=bool)

    flat_targets = targets.reshape(-1)
    flat_mask = scored_mask.reshape(-1)
    flat_logits = logits.reshape(-1, class_count)

    scored_targets = flat_targets[flat_mask]
    scored_logits = flat_logits[flat_mask]
    scored_count = int(scored_targets.size)
    ignored_count = int(flat_targets.size - scored_count)

    if scored_count == 0:
        return {
            "k": k,
            "correct_count": 0,
            "scored_count": 0,
            "ignored_count": ignored_count,
            "accuracy": None,
        }

    if np.any(scored_targets < 0) or np.any(scored_targets >= class_count):
        raise ValueError("scored targets must be valid class indices")
    if not np.all(np.isfinite(scored_logits)):
        raise ValueError("scored logits must contain only finite values")

    # Stable sorting makes exact ties reproducible: lower class ids win because the
    # original class axis is already in ascending index order.
    ranking = np.argsort(-scored_logits, axis=-1, kind="stable")[:, :k]
    correct = np.any(ranking == scored_targets[:, None], axis=1)
    correct_count = int(np.count_nonzero(correct))

    return {
        "k": k,
        "correct_count": correct_count,
        "scored_count": scored_count,
        "ignored_count": ignored_count,
        "accuracy": float(correct_count / scored_count),
    }


def topk_accuracy(logits, targets, *, k=1, ignore_index=None):
    """Return only the top-k accuracy, or ``None`` when every target is ignored."""
    return topk_accuracy_report(
        logits,
        targets,
        k=k,
        ignore_index=ignore_index,
    )["accuracy"]
