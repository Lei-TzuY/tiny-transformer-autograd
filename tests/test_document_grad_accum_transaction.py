"""Failure transactionality for ragged document gradient accumulation."""

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


def test_late_unscored_microbatch_restores_preexisting_gradients_exactly():
    model = _tiny_model(seed=11)
    params = tuple(model.parameters())
    valid = (
        np.array([[1, 2, 0]], dtype=np.int64),
        np.array([[2, 3, train.IGNORE_INDEX]], dtype=np.int64),
        np.array([[1, 1, 0]], dtype=np.int64),
    )
    invalid = (
        np.array([[3, 0, 0]], dtype=np.int64),
        np.full((1, 3), train.IGNORE_INDEX, dtype=np.int64),
        np.zeros((1, 3), dtype=np.int64),
    )

    original_buffers = []
    original_values = []
    for index, parameter in enumerate(params):
        parameter.grad = np.full_like(parameter.data, 0.25 + index * 0.01)
        original_buffers.append(parameter.grad)
        original_values.append(parameter.grad.copy())

    batches = iter([valid, invalid])
    with pytest.raises(ValueError, match="no scored tokens"):
        train.accumulate_document_gradients(
            model,
            lambda: next(batches),
            params,
            grad_accum=2,
        )

    for parameter, buffer, expected in zip(params, original_buffers, original_values):
        assert parameter.grad is buffer
        np.testing.assert_array_equal(parameter.grad, expected)


def test_late_sampler_failure_restores_none_and_array_gradient_states():
    model = _tiny_model(seed=12)
    params = tuple(model.parameters())
    valid = (
        np.array([[1, 2, 0]], dtype=np.int64),
        np.array([[2, 3, train.IGNORE_INDEX]], dtype=np.int64),
        np.array([[1, 1, 0]], dtype=np.int64),
    )

    original = []
    for index, parameter in enumerate(params):
        if index % 2 == 0:
            parameter.grad = None
            original.append((None, None))
        else:
            parameter.grad = np.full_like(parameter.data, index + 0.5)
            original.append((parameter.grad, parameter.grad.copy()))

    calls = 0

    def sample_batch():
        nonlocal calls
        calls += 1
        if calls == 1:
            return valid
        raise RuntimeError("late sampler failure")

    with pytest.raises(RuntimeError, match="late sampler failure"):
        train.accumulate_document_gradients(model, sample_batch, params, grad_accum=2)

    for parameter, (buffer, expected) in zip(params, original):
        if buffer is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is buffer
            np.testing.assert_array_equal(parameter.grad, expected)
