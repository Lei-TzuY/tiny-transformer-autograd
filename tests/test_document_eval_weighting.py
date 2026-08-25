"""Regression tests for scored-token-weighted document validation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.tensor import Tensor
import train as train_module
from train import IGNORE_INDEX, evaluate_batches, evaluate_documents


class _ModeOnlyModel:
    def __init__(self, training=True):
        self.training = training

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


def _batches():
    one_scored = (
        np.zeros((1, 3), dtype=np.int64),
        np.array([[1, IGNORE_INDEX, IGNORE_INDEX]], dtype=np.int64),
        np.array([[1, 0, 0]], dtype=np.int64),
    )
    three_scored = (
        np.zeros((1, 3), dtype=np.int64),
        np.array([[1, 1, 1]], dtype=np.int64),
        np.array([[1, 1, 1]], dtype=np.int64),
    )
    return one_scored, three_scored


def _fake_batch_loss(_model, _tokens, targets, _mask=None):
    scored = int(np.count_nonzero(targets != IGNORE_INDEX))
    return Tensor(2.0 if scored == 1 else 4.0)


def _sampler(batches):
    iterator = iter(batches)
    return lambda: next(iterator)


def test_weighted_batches_use_the_cross_entropy_scored_count(monkeypatch):
    monkeypatch.setattr(train_module, "batch_loss", _fake_batch_loss)
    model = _ModeOnlyModel()
    batches = _batches()

    weighted_loss, weighted_ppl = evaluate_batches(
        model,
        _sampler(batches),
        eval_iters=2,
        weight_by_scored_tokens=True,
    )

    # Per-batch means are 2 and 4, but they summarize 1 and 3 scored tokens.
    # The corpus loss is therefore (2*1 + 4*3) / 4 = 3.5, not mean([2, 4]).
    assert weighted_loss == pytest.approx(3.5)
    assert weighted_ppl == pytest.approx(np.exp(3.5))
    assert weighted_loss != pytest.approx(3.0)
    assert model.training is True


def test_document_evaluation_opts_into_token_weighting(monkeypatch):
    monkeypatch.setattr(train_module, "batch_loss", _fake_batch_loss)
    batches = iter(_batches())
    monkeypatch.setattr(
        train_module,
        "get_document_batch",
        lambda _documents, _batch_size: next(batches),
    )

    loss, perplexity = evaluate_documents(
        _ModeOnlyModel(),
        documents=[np.array([0, 1], dtype=np.int64)],
        batch_size=1,
        eval_iters=2,
    )

    assert loss == pytest.approx(3.5)
    assert perplexity == pytest.approx(np.exp(3.5))


def test_unweighted_evaluation_preserves_batch_mean_semantics(monkeypatch):
    monkeypatch.setattr(train_module, "batch_loss", _fake_batch_loss)

    loss, _ = evaluate_batches(
        _ModeOnlyModel(),
        _sampler(_batches()),
        eval_iters=2,
    )

    assert loss == pytest.approx(3.0)
