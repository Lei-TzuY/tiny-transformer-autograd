"""Regression tests for token-weighted document gradient accumulation."""

import os
import sys

import numpy as np

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
    accumulated_loss = train.accumulate_gradients(
        model,
        model.parameters(),
        _sampler([short, long]),
        grad_accum=2,
        weight_by_scored_tokens=True,
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


def test_equal_microbatch_path_preserves_historical_accumulation_math():
    """The default token-stream path keeps the previous operation sequence."""
    model = _tiny_model(seed=2)
    batch_a = (
        np.array([[0, 1, 2]], dtype=np.int64),
        np.array([[1, 2, 3]], dtype=np.int64),
        None,
    )
    batch_b = (
        np.array([[3, 4, 5]], dtype=np.int64),
        np.array([[4, 5, 6]], dtype=np.int64),
        None,
    )

    model.zero_grad()
    actual_loss = train.accumulate_gradients(
        model,
        model.parameters(),
        _sampler([batch_a, batch_b]),
        grad_accum=2,
        weight_by_scored_tokens=False,
    )
    actual_grads = _grads(model)

    model.zero_grad()
    losses = []
    for batch in (batch_a, batch_b):
        loss = train.batch_loss(model, *batch)
        loss.backward()
        losses.append(float(loss.data))
    for parameter in model.parameters():
        parameter.grad /= 2

    assert actual_loss == float(np.mean(losses))
    for name, parameter in model.named_parameters():
        np.testing.assert_array_equal(actual_grads[name], parameter.grad)


def test_single_document_microbatch_keeps_direct_backward_path():
    """Token weighting is a no-op when there is only one micro-batch."""
    model = _tiny_model(seed=3)
    short, _ = _ragged_microbatches()

    model.zero_grad()
    actual_loss = train.accumulate_gradients(
        model,
        model.parameters(),
        _sampler([short]),
        grad_accum=1,
        weight_by_scored_tokens=True,
    )
    actual_grads = _grads(model)

    model.zero_grad()
    direct_loss = train.batch_loss(model, *short)
    direct_loss.backward()

    assert actual_loss == float(direct_loss.data)
    for name, parameter in model.named_parameters():
        np.testing.assert_array_equal(actual_grads[name], parameter.grad)
