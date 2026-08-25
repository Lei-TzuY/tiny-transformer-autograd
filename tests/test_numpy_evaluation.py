"""Parity and contract tests for NumPy-only validation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from engine.grad_mode import no_grad
from engine.tensor import Tensor
from nn.transformer import GPT
from train import (
    IGNORE_INDEX,
    _cross_entropy_np,
    batch_eval_loss,
    batch_loss,
    evaluate_batches,
)


def test_numpy_cross_entropy_matches_graph_loss():
    logits = np.array(
        [
            [[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]],
            [[-0.5, 0.25, 1.5], [3.0, 2.0, 1.0]],
        ],
        dtype=np.float64,
    )
    targets = np.array([[0, 1], [2, 0]], dtype=np.int64)

    graph = float(ops.cross_entropy(Tensor(logits), targets).data)
    fast = _cross_entropy_np(logits, targets)

    assert fast == pytest.approx(graph, rel=1e-14, abs=1e-14)


def test_numpy_cross_entropy_matches_ignore_index_semantics():
    logits = np.array(
        [[[2.0, 0.0], [-np.inf, -np.inf], [0.5, 1.5]]],
        dtype=np.float64,
    )
    targets = np.array([[0, IGNORE_INDEX, 1]], dtype=np.int64)

    graph = float(
        ops.cross_entropy(Tensor(logits), targets, ignore_index=IGNORE_INDEX).data
    )
    fast = _cross_entropy_np(logits, targets, ignore_index=IGNORE_INDEX)

    assert fast == pytest.approx(graph, rel=1e-14, abs=1e-14)


def test_numpy_cross_entropy_rejects_invalid_scored_rows():
    logits = np.array([[[-np.inf, -np.inf]]], dtype=np.float64)
    targets = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="at least one finite logit"):
        _cross_entropy_np(logits, targets)

    with pytest.raises(ValueError, match="no scored target"):
        _cross_entropy_np(logits, np.array([[IGNORE_INDEX]]), ignore_index=IGNORE_INDEX)


@pytest.mark.parametrize(
    "architecture",
    [
        {"norm": "layernorm", "pos_encoding": "learned", "ffn": "gelu"},
        {"norm": "rmsnorm", "pos_encoding": "rope", "ffn": "swiglu"},
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_batch_eval_loss_matches_graph_forward(architecture, masked):
    np.random.seed(123)
    model = GPT(
        vocab_size=7,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.35,
        **architecture,
    )
    model.eval()

    tokens = np.array([[1, 2, 3, 4], [4, 3, 0, 0]], dtype=np.int64)
    if masked:
        mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int64)
        targets = np.array(
            [[2, 3, 4, 5], [3, 2, IGNORE_INDEX, IGNORE_INDEX]],
            dtype=np.int64,
        )
    else:
        mask = None
        targets = np.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=np.int64)

    with no_grad():
        graph = float(batch_loss(model, tokens, targets, mask).data)
    fast = batch_eval_loss(model, tokens, targets, mask)

    assert fast == pytest.approx(graph, rel=1e-10, abs=1e-12)


class _InferOnlyModel:
    def __init__(self, fail=False):
        self.training = True
        self.fail = fail
        self.infer_calls = 0

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("graph forward must not run on the infer fast path")

    def infer(self, tokens, attention_mask=None):
        self.infer_calls += 1
        if self.fail:
            raise RuntimeError("inference failed")
        logits = np.zeros((*np.asarray(tokens).shape, 3), dtype=np.float64)
        return logits, None


def test_evaluate_batches_prefers_infer_over_graph_forward():
    model = _InferOnlyModel()
    batch = (
        np.array([[0, 1]], dtype=np.int64),
        np.array([[1, 2]], dtype=np.int64),
        None,
    )

    loss, perplexity = evaluate_batches(model, lambda: batch, eval_iters=3)

    assert loss == pytest.approx(np.log(3.0))
    assert perplexity == pytest.approx(3.0)
    assert model.infer_calls == 3
    assert model.training is True


def test_fast_evaluation_restores_mode_after_infer_failure():
    model = _InferOnlyModel(fail=True)
    batch = (
        np.array([[0, 1]], dtype=np.int64),
        np.array([[1, 2]], dtype=np.int64),
        None,
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate_batches(model, lambda: batch, eval_iters=1)

    assert model.training is True


def test_fast_evaluation_preserves_right_padding_contract():
    np.random.seed(7)
    model = GPT(
        vocab_size=5,
        context_len=3,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    )
    tokens = np.array([[0, 1, 2]], dtype=np.int64)
    targets = np.array([[IGNORE_INDEX, 2, 3]], dtype=np.int64)
    left_padded = np.array([[0, 1, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="right-padded"):
        batch_eval_loss(model, tokens, targets, left_padded)
