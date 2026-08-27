"""Training CLI integration tests for label-smoothed cross entropy."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
import train
from engine import Tensor, label_smoothed_cross_entropy
from nn.transformer import GPT


class _StaticModel:
    def __init__(self, logits):
        self.logits = Tensor(logits, requires_grad=True)
        self.calls = []

    def __call__(self, tokens, attention_mask=None):
        self.calls.append(None if attention_mask is None else np.array(attention_mask, copy=True))
        return self.logits


class _MetadataModel:
    def config(self):
        return {"vocab_size": 3, "context_len": 2}


class _MetadataTokenizer:
    def state_dict(self):
        return {"kind": "char", "tokens": ["a", "b", "c"]}


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
    return {name: parameter.grad.copy() for name, parameter in model.named_parameters()}


def test_batch_loss_zero_smoothing_matches_historical_cross_entropy_value_and_vjp():
    logits_data = np.array(
        [[[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]]], dtype=np.float64
    )
    targets = np.array([[2, 1]], dtype=np.int64)
    tokens = np.array([[0, 1]], dtype=np.int64)

    model = _StaticModel(logits_data)
    loss = train.batch_loss(model, tokens, targets, label_smoothing=0.0)
    loss.backward()

    reference_logits = Tensor(logits_data, requires_grad=True)
    reference = ops.cross_entropy(reference_logits, targets)
    reference.backward()

    np.testing.assert_array_equal(loss.data, reference.data)
    np.testing.assert_array_equal(model.logits.grad, reference_logits.grad)
    assert model.calls == [None]


def test_batch_loss_positive_smoothing_matches_engine_primitive():
    logits_data = np.array(
        [[[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]]], dtype=np.float64
    )
    targets = np.array([[2, 1]], dtype=np.int64)
    tokens = np.array([[0, 1]], dtype=np.int64)

    model = _StaticModel(logits_data)
    loss = train.batch_loss(model, tokens, targets, label_smoothing=0.2)
    loss.backward()

    reference_logits = Tensor(logits_data, requires_grad=True)
    reference = label_smoothed_cross_entropy(
        reference_logits, targets, smoothing=0.2
    )
    reference.backward()

    np.testing.assert_allclose(loss.data, reference.data, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(model.logits.grad, reference_logits.grad, atol=0.0, rtol=0.0)


def test_masked_batch_label_smoothing_preserves_ignore_index_contract():
    logits_data = np.array(
        [
            [[2.0, 0.0, -1.0], [np.nan, np.inf, -np.inf], [0.5, 1.0, -0.5]],
        ],
        dtype=np.float64,
    )
    tokens = np.array([[1, 0, 2]], dtype=np.int64)
    targets = np.array([[0, train.IGNORE_INDEX, 1]], dtype=np.int64)
    mask = np.array([[1, 0, 1]], dtype=np.int64)
    model = _StaticModel(logits_data)

    loss = train.batch_loss(
        model,
        tokens,
        targets,
        mask,
        label_smoothing=0.1,
    )
    loss.backward()

    assert np.isfinite(loss.data)
    np.testing.assert_array_equal(model.logits.grad[0, 1], np.zeros(3))
    np.testing.assert_array_equal(model.calls[0], mask)


def test_batch_loss_rejects_invalid_smoothing_before_model_forward():
    class NeverCalled:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("model must not run")

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        train.batch_loss(
            NeverCalled(),
            np.array([[0]]),
            np.array([[0]]),
            label_smoothing=-0.1,
        )
    with pytest.raises(TypeError, match="real number"):
        train.batch_loss(
            NeverCalled(),
            np.array([[0]]),
            np.array([[0]]),
            label_smoothing=True,
        )


def test_document_accumulation_with_smoothing_matches_token_weighted_large_batch():
    model = _tiny_model(seed=11)
    short, long = _ragged_microbatches()

    model.zero_grad()
    accumulated_loss = train.accumulate_document_gradients(
        model,
        _sampler([short, long]),
        model.parameters(),
        grad_accum=2,
        label_smoothing=0.2,
    )
    accumulated_grads = _grads(model)

    combined = tuple(np.concatenate([a, b], axis=0) for a, b in zip(short, long))
    model.zero_grad()
    large_loss = train.batch_loss(
        model, *combined, label_smoothing=0.2
    )
    large_loss.backward()

    np.testing.assert_allclose(accumulated_loss, float(large_loss.data), atol=1e-12)
    for name, parameter in model.named_parameters():
        np.testing.assert_allclose(
            accumulated_grads[name], parameter.grad, atol=1e-10, rtol=1e-10
        )


def test_validation_remains_unsmoothed_when_training_objective_is_smoothed():
    logits_data = np.array([[[4.0, 1.0, -2.0]]], dtype=np.float64)
    targets = np.array([[0]], dtype=np.int64)
    tokens = np.array([[0]], dtype=np.int64)
    model = _StaticModel(logits_data)

    training_loss = float(
        train.batch_loss(model, tokens, targets, label_smoothing=0.4).data
    )
    validation_loss = train.batch_eval_loss(model, tokens, targets)
    expected_validation = train._cross_entropy_np(logits_data, targets)

    assert training_loss != pytest.approx(validation_loss)
    assert validation_loss == pytest.approx(expected_validation, abs=0.0)


def test_metadata_keeps_legacy_shape_when_smoothing_is_zero():
    metadata = train._metadata(_MetadataModel(), _MetadataTokenizer(), 0.0)

    assert set(metadata) == {"model_config", "tokenizer"}
    assert "training_config" not in metadata


def test_metadata_records_nonzero_label_smoothing():
    metadata = train._metadata(_MetadataModel(), _MetadataTokenizer(), 0.15)

    assert metadata["training_config"] == {"label_smoothing": 0.15}


def test_label_smoothing_resolution_defaults_and_inherits_checkpoint_value():
    assert train._resolve_label_smoothing(None, {}) == 0.0
    assert train._resolve_label_smoothing(None, {"model_config": {}}) == 0.0

    metadata = {"training_config": {"label_smoothing": 0.25}}
    assert train._resolve_label_smoothing(None, metadata) == 0.25
    assert train._resolve_label_smoothing(0.25, metadata) == 0.25


def test_label_smoothing_resolution_rejects_objective_change_on_resume():
    metadata = {"training_config": {"label_smoothing": 0.25}}

    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        train._resolve_label_smoothing(0.1, metadata)
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        train._resolve_label_smoothing(0.0, metadata)


@pytest.mark.parametrize(
    "metadata, error, message",
    [
        ({"training_config": []}, ValueError, "must be a mapping"),
        ({"training_config": {"label_smoothing": True}}, TypeError, "real number"),
        ({"training_config": {"label_smoothing": np.nan}}, ValueError, "finite"),
        ({"training_config": {"label_smoothing": 1.1}}, ValueError, r"\[0, 1\]"),
    ],
)
def test_label_smoothing_resolution_validates_checkpoint_metadata(metadata, error, message):
    with pytest.raises(error, match=message):
        train._resolve_label_smoothing(None, metadata)


def test_parse_args_distinguishes_unspecified_from_explicit_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiny-train"])
    assert train.parse_args().label_smoothing is None

    monkeypatch.setattr(sys, "argv", ["tiny-train", "--label-smoothing", "0"])
    assert train.parse_args().label_smoothing == 0.0


def test_validate_args_rejects_invalid_cli_label_smoothing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiny-train", "--label-smoothing", "nan"])
    with pytest.raises(ValueError, match="finite"):
        train._validate_args(train.parse_args())

    monkeypatch.setattr(sys, "argv", ["tiny-train", "--label-smoothing", "1.01"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        train._validate_args(train.parse_args())


def test_metadata_and_resolution_round_trip_training_objective():
    metadata = train._metadata(_MetadataModel(), _MetadataTokenizer(), 0.3)

    assert train._resolve_label_smoothing(None, metadata) == 0.3
