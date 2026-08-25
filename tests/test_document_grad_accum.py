"""Regression tests for token-weighted document gradient accumulation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train
from nn.transformer import GPT


def _tiny_model(seed=0):
    np.random.seed(seed)
    return GPT(
        vocab_size=7,
        context_len=3,
        d_model=4,
        num_heads=1,
        d_ff=8,
        num_layers=1,
        dropout=0.0,
    )


def _ragged_microbatches():
    short = (
        np.array([[1, 0, 0]], dtype=np.int64),
        np.array([[2, train.IGNORE_INDEX, train.IGNORE_INDEX]], dtype=np.int64),
        np.array([[1, 0, 0]], dtype=np.int64),
    )
    long = (
        np.array([[3, 4, 5]], dtype=np.int64),
        np.array([[4, 5, 6]], dtype=np.int64),
        np.array([[1, 1, 1]], dtype=np.int64),
    )
    return short, long


def _sampler(batches):
    iterator = iter(batches)
    return lambda: next(iterator)


def _grads(model):
    return {
        name: parameter.grad.copy()
        for name, parameter in model.named_parameters()
    }


def test_document_accumulation_matches_one_token_weighted_large_batch():
    """Ragged micro-batches must equal one batch over all scored tokens."""
    model = _tiny_model(seed=1)
    short, long = _ragged_microbatches()

    model.zero_grad()
    accumulated_loss = train.accumulate_document_gradients(
        model,
        _sampler([short, long]),
        model.parameters(),
        grad_accum=2,
    )
    accumulated_grads = _grads(model)

    combined = tuple(np.concatenate([a, b], axis=0) for a, b in zip(short, long))
    model.zero_grad()
    large_loss = train.batch_loss(model, *combined)
    large_loss.backward()

    np.testing.assert_allclose(accumulated_loss, float(large_loss.data), atol=1e-12)
    for name, parameter in model.named_parameters():
        np.testing.assert_allclose(
            accumulated_grads[name], parameter.grad, atol=1e-10, rtol=1e-10
        )


def test_equal_microbatch_count_is_not_used_for_ragged_losses():
    """Pin the old mean-of-means bug with deliberately unequal token counts."""
    model = _tiny_model(seed=2)
    short, long = _ragged_microbatches()

    individual_losses = [
        float(train.batch_loss(model, *batch).data)
        for batch in (short, long)
    ]
    model.zero_grad()
    weighted = train.accumulate_document_gradients(
        model,
        _sampler([short, long]),
        model.parameters(),
        grad_accum=2,
    )

    expected = (individual_losses[0] + 3.0 * individual_losses[1]) / 4.0
    assert weighted == pytest.approx(expected, abs=1e-12)
    assert weighted != pytest.approx(float(np.mean(individual_losses)), abs=1e-8)


def test_document_accumulation_is_only_used_for_multiple_microbatches():
    model = _tiny_model(seed=3)
    short, _ = _ragged_microbatches()

    with pytest.raises(ValueError, match="grad_accum > 1"):
        train.accumulate_document_gradients(
            model,
            _sampler([short]),
            model.parameters(),
            grad_accum=1,
        )


def test_unscored_document_microbatch_fails_before_backward():
    model = _tiny_model(seed=4)
    bad = (
        np.array([[1, 0, 0]], dtype=np.int64),
        np.full((1, 3), train.IGNORE_INDEX, dtype=np.int64),
        np.zeros((1, 3), dtype=np.int64),
    )
    model.zero_grad()
    before = _grads(model)

    with pytest.raises(ValueError, match="no scored tokens"):
        train.accumulate_document_gradients(
            model,
            _sampler([bad, bad]),
            model.parameters(),
            grad_accum=2,
        )

    after = _grads(model)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
