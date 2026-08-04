"""Public API validation and failure-mode regression tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.optim import Adam, SGD
from engine.tensor import Tensor
from nn.attention import MultiHeadAttention, RotaryEmbedding, _causal_mask
from nn.layers import Dropout, Embedding, LayerNorm, RMSNorm
from nn.transformer import GPT


def _model(**overrides):
    config = dict(
        vocab_size=8,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    )
    config.update(overrides)
    return GPT(**config)


@pytest.mark.parametrize("probability", [-0.1, 1.0, 2.0])
def test_dropout_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="dropout probability"):
        Dropout(probability)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": 0.0}, "lr"),
        ({"betas": (1.0, 0.999)}, "betas"),
        ({"betas": (0.9,)}, "betas"),
        ({"eps": 0.0}, "eps"),
        ({"weight_decay": -0.1}, "weight_decay"),
    ],
)
def test_adam_rejects_invalid_hyperparameters(kwargs, message):
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match=message):
        Adam([parameter], **kwargs)


def test_sgd_materializes_parameter_generators_and_validates_momentum():
    parameters = [Tensor([1.0], requires_grad=True), Tensor([2.0], requires_grad=True)]
    optimizer = SGD((parameter for parameter in parameters), lr=0.1)
    assert len(optimizer.parameters) == 2
    with pytest.raises(ValueError, match="momentum"):
        SGD(parameters, momentum=1.0)


def test_attention_dimensions_raise_value_errors():
    with pytest.raises(ValueError, match="positive"):
        MultiHeadAttention(d_model=8, num_heads=0)
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=7, num_heads=2)


def test_rope_bounds_and_dimension_are_validated():
    rope = RotaryEmbedding(dim=4, max_pos=3)
    x = Tensor(np.zeros((1, 1, 2, 4)))
    with pytest.raises(ValueError, match="exceed"):
        rope.rotate(x, offset=2)
    with pytest.raises(ValueError, match="end in dimension"):
        rope.rotate_np(np.zeros((1, 2, 6)))


def test_causal_mask_is_impenetrable_negative_infinity():
    mask = _causal_mask(query_len=3, key_len=3, past_len=0)
    assert np.isneginf(mask[0, 1])
    assert mask[2, 2] == 0.0


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((2, 2, 4, 4)), "larger than"),
        (np.zeros((5, 5)), "does not broadcast"),
        (np.zeros((4, 3)), "does not broadcast"),
        (np.full((4, 4), np.nan), "finite biases"),
        (np.full((4, 4), np.inf), "finite biases"),
    ],
)
def test_attention_rejects_malformed_custom_masks(mask, message):
    attention = MultiHeadAttention(d_model=8, num_heads=2)
    x = Tensor(np.zeros((1, 4, 8)))
    with pytest.raises(ValueError, match=message):
        attention(x, mask)


def test_attention_accepts_negative_infinity_masks():
    attention = MultiHeadAttention(d_model=8, num_heads=2)
    x = Tensor(np.zeros((1, 4, 8)))
    out = attention(x, np.triu(np.full((4, 4), -np.inf), k=1))
    assert np.isfinite(out.data).all()


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([0, 1]),
        np.empty((1, 0), dtype=np.int64),
        np.array([[0, -1]]),
        np.array([[0, 8]]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_gpt_rejects_invalid_token_batches(tokens):
    with pytest.raises((TypeError, ValueError)):
        _model().forward(tokens)


@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (np.ones((1, 3), dtype=np.int64), ValueError, "matching the token ids"),
        (np.ones((2, 2), dtype=np.int64), ValueError, "matching the token ids"),
        (np.array([[1, 2]]), ValueError, "0/False"),
        (np.array([[1, -1]]), ValueError, "0/False"),
        (np.array([["1", "0"]]), TypeError, "boolean or numeric"),
    ],
)
def test_gpt_rejects_malformed_attention_masks(mask, error, message):
    tokens = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(error, match=message):
        _model().forward(tokens, attention_mask=mask)


def test_gpt_accepts_a_well_formed_attention_mask():
    tokens = np.array([[0, 1]], dtype=np.int64)
    logits = _model().forward(tokens, attention_mask=np.array([[1, 0]]))
    assert logits.shape == (1, 2, 8)
    assert np.isfinite(logits.data).all()


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[0, 1, 1, 1]]),
        np.array([[1, 0, 1, 0]]),
    ],
)
def test_gpt_forward_rejects_left_or_interior_padding(mask):
    tokens = np.array([[0, 1, 2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="right-padded"):
        _model().forward(tokens, attention_mask=mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[1, 1, 1, 1]]),
        np.array([[1, 1, 0, 0]]),
        np.array([[0, 0, 0, 0]]),
    ],
)
def test_gpt_forward_accepts_right_padding_and_all_padding(mask):
    tokens = np.array([[0, 1, 2, 3]], dtype=np.int64)
    logits = _model().forward(tokens, attention_mask=mask)
    assert logits.shape == (1, 4, 8)
    assert np.isfinite(logits.data).all()


def test_infer_attention_mask_must_cover_cached_keys():
    model = _model()
    tokens = np.array([[0, 1]], dtype=np.int64)
    _, cache = model.infer(tokens)

    with pytest.raises(ValueError, match=r"\(1, 3\) covering the cached"):
        model.infer(np.array([[2]]), cache, attention_mask=np.ones((1, 1)))

    logits, _ = model.infer(
        np.array([[2]]), cache, attention_mask=np.ones((1, 3), dtype=np.int64)
    )
    assert np.isfinite(logits).all()


@pytest.mark.parametrize(
    ("cache", "error", "message"),
    [
        ({}, TypeError, "list or tuple"),
        ([], ValueError, "one entry per transformer block"),
        ([None], TypeError, "dictionary"),
        ([{"k": np.zeros((1, 2, 1, 4))}], ValueError, "'k' and 'v'"),
        (
            [{"k": [[[[0.0] * 4]] * 2], "v": [[[[0.0] * 4]] * 2]}],
            TypeError,
            "NumPy arrays",
        ),
        (
            [{"k": np.zeros((1, 2, 4)), "v": np.zeros((1, 2, 4))}],
            ValueError,
            "rank 4",
        ),
        (
            [
                {
                    "k": np.zeros((1, 2, 1, 4)),
                    "v": np.zeros((1, 2, 2, 4)),
                }
            ],
            ValueError,
            "equal shapes",
        ),
        (
            [{"k": np.zeros((2, 2, 1, 4)), "v": np.zeros((2, 2, 1, 4))}],
            ValueError,
            "batch dimension",
        ),
        (
            [{"k": np.zeros((1, 1, 1, 4)), "v": np.zeros((1, 1, 1, 4))}],
            ValueError,
            "head count",
        ),
        (
            [{"k": np.zeros((1, 2, 1, 3)), "v": np.zeros((1, 2, 1, 3))}],
            ValueError,
            "head dimension",
        ),
        (
            [
                {
                    "k": np.zeros((1, 2, 1, 4)).astype(object),
                    "v": np.zeros((1, 2, 1, 4)).astype(object),
                }
            ],
            TypeError,
            "real numeric",
        ),
        (
            [
                {
                    "k": np.full((1, 2, 1, 4), np.nan),
                    "v": np.zeros((1, 2, 1, 4)),
                }
            ],
            ValueError,
            "finite values",
        ),
    ],
)
def test_infer_rejects_malformed_kv_cache(cache, error, message):
    with pytest.raises(error, match=message):
        _model().infer(np.array([[0]], dtype=np.int64), kv_cache=cache)


def test_infer_rejects_inconsistent_cache_lengths_across_layers():
    model = _model(num_layers=2)
    cache = [
        {"k": np.zeros((1, 2, 1, 4)), "v": np.zeros((1, 2, 1, 4))},
        {"k": np.zeros((1, 2, 2, 4)), "v": np.zeros((1, 2, 2, 4))},
    ]
    with pytest.raises(ValueError, match="same past length"):
        model.infer(np.array([[0]], dtype=np.int64), kv_cache=cache)


def test_infer_accepts_tuple_cache_and_preserves_existing_entries():
    model = _model()
    _, cache = model.infer(np.array([[0, 1]], dtype=np.int64))
    original_key = cache[0]["k"].copy()
    original_value = cache[0]["v"].copy()

    logits, updated = model.infer(np.array([[2]], dtype=np.int64), tuple(cache))

    assert logits.shape == (1, 1, 8)
    assert updated[0]["k"].shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(cache[0]["k"], original_key)
    np.testing.assert_array_equal(cache[0]["v"], original_value)


@pytest.mark.parametrize(
    ("position_ids", "error"),
    [
        (np.array([[0, 1, 2]]), ValueError),
        (np.array([[0.0, 1.0]]), TypeError),
        (np.array([[0, 9]]), ValueError),
        (np.array([[-1, 0]]), ValueError),
    ],
)
def test_infer_rejects_malformed_position_ids(position_ids, error):
    with pytest.raises(error):
        _model().infer(np.array([[0, 1]], dtype=np.int64), position_ids=position_ids)


def test_rope_positions_are_validated():
    rope = RotaryEmbedding(dim=4, max_pos=4)
    x = np.zeros((2, 1, 3, 4))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        rope.rotate_np(x, positions=np.array([[[0, 1, 4]], [[0, 1, 2]]]))
    with pytest.raises(TypeError, match="integers"):
        rope.rotate_np(x, positions=np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="larger than"):
        rope.rotate_np(x, positions=np.zeros((2, 5, 3), dtype=np.int64))


def test_rope_positions_match_a_uniform_offset():
    rope = RotaryEmbedding(dim=4, max_pos=8)
    x = np.arange(12.0).reshape(1, 1, 3, 4)
    offset = rope.rotate_np(x, offset=2)
    explicit = rope.rotate_np(x, positions=np.array([[[2, 3, 4]]]))
    np.testing.assert_array_equal(offset, explicit)


def test_gpt_rejects_sequence_beyond_context():
    with pytest.raises(ValueError, match="context_len"):
        _model().forward(np.zeros((1, 5), dtype=np.int64))


def test_generation_rejects_invalid_length_and_empty_prompt():
    model = _model()
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(np.array([[0]], dtype=np.int64), -1)
    with pytest.raises(ValueError, match="empty"):
        model.generate(np.empty((1, 0), dtype=np.int64), 1)


def test_embedding_rejects_negative_indices_in_forward_and_infer():
    embedding = Embedding(4, 2)
    with pytest.raises(ValueError, match="indices"):
        embedding(np.array([-1]))
    with pytest.raises(ValueError, match="indices"):
        embedding.infer(np.array([-1]))


@pytest.mark.parametrize("norm_cls", [LayerNorm, RMSNorm])
def test_norms_reject_mismatched_final_dimension(norm_cls):
    norm = norm_cls(4)
    with pytest.raises(ValueError, match="final dimension"):
        norm(Tensor(np.zeros((2, 3))))
    with pytest.raises(ValueError, match="final dimension"):
        norm.infer(np.zeros((2, 3)))
