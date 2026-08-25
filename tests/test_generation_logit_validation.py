"""Regression tests for generation-time logit validation and stability."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.transformer import GPT, _log_softmax, _sample


def _tiny_model(vocab_size=3):
    return GPT(
        vocab_size=vocab_size,
        context_len=4,
        d_model=4,
        num_heads=2,
        d_ff=8,
        num_layers=1,
        dropout=0.0,
    )


def _install_logits(model, rows):
    rows = np.asarray(rows)

    def infer(idx, kv_cache=None, attention_mask=None, position_ids=None):
        batch, time = idx.shape
        logits = np.zeros((batch, time, rows.shape[-1]), dtype=rows.dtype)
        if rows.ndim == 1:
            logits[:, -1, :] = rows
        else:
            logits[:, -1, :] = rows[:batch]
        return logits, []

    model.infer = infer


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize(
    "logits",
    [
        np.array([0.0, np.nan, -1.0]),
        np.array([0.0, np.inf, -1.0]),
        np.array([-np.inf, -np.inf, -np.inf]),
    ],
)
def test_sample_rejects_invalid_logits_without_consuming_rng(logits):
    np.random.seed(314159)
    before = np.random.get_state()

    with pytest.raises(ValueError, match=r"NaN|\+inf|finite"):
        _sample(logits)

    after = np.random.get_state()
    _assert_rng_state_equal(after, before)


@pytest.mark.parametrize(
    "logits",
    [
        np.array([True, False]),
        np.array([1.0 + 2.0j, 0.0 + 0.0j]),
        np.array(["1", "2"]),
    ],
)
def test_sample_rejects_non_real_numeric_logits(logits):
    with pytest.raises(TypeError, match="real numeric dtype"):
        _sample(logits)


def test_partial_negative_infinity_remains_a_valid_impossible_token():
    for seed in range(5):
        np.random.seed(seed)
        assert _sample(np.array([0.0, -np.inf, -np.inf])) == 0

    log_probs = _log_softmax(np.array([0.0, -np.inf, -1.0]))
    assert np.isfinite(log_probs[[0, 2]]).all()
    assert np.isneginf(log_probs[1])


def test_tiny_temperature_stabilizes_large_finite_logits_without_warning():
    # Direct historical division overflows both entries. The largest class is
    # still mathematically unambiguous, so selection should remain well-defined.
    np.random.seed(7)
    token = _sample(np.array([1e308, -1e308]), temperature=1e-300)
    assert token == 0


def test_ordinary_seeded_sampling_keeps_historical_arithmetic_result():
    logits = np.array([0.2, -0.5, 1.1, 0.7], dtype=np.float64)
    temperature = 0.7
    top_k = 3
    top_p = 0.85

    def historical_sample():
        filtered = np.array(logits, dtype=np.float64, copy=True) / temperature
        threshold = np.partition(filtered, -top_k)[-top_k]
        filtered[filtered < threshold] = -np.inf
        order = np.argsort(filtered)[::-1]
        sorted_logits = filtered[order]
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs /= sorted_probs.sum()
        remove = np.cumsum(sorted_probs) > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        filtered[order[remove]] = -np.inf
        probs = np.exp(filtered - np.max(filtered))
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    np.random.seed(2026)
    expected = historical_sample()
    np.random.seed(2026)
    actual = _sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    assert actual == expected


@pytest.mark.parametrize(
    "bad_row",
    [
        [0.0, np.nan, -1.0],
        [0.0, np.inf, -1.0],
        [-np.inf, -np.inf, -np.inf],
    ],
)
def test_greedy_generation_rejects_invalid_model_logits_before_argmax(
    monkeypatch, bad_row
):
    model = _tiny_model()
    _install_logits(model, np.array(bad_row))

    def forbidden_argmax(*args, **kwargs):
        raise AssertionError("argmax must not see invalid generation logits")

    monkeypatch.setattr(np, "argmax", forbidden_argmax)
    with pytest.raises(ValueError, match=r"NaN|\+inf|finite"):
        model.generate(
            np.array([[0]], dtype=np.int64),
            max_new_tokens=1,
            strategy="greedy",
            use_cache=False,
        )


def test_batched_greedy_rejects_one_bad_row_before_selecting_any_tokens():
    model = _tiny_model()
    _install_logits(
        model,
        np.array(
            [
                [0.0, 1.0, -np.inf],
                [0.0, np.nan, -1.0],
            ]
        ),
    )

    with pytest.raises(ValueError, match="NaN"):
        model.generate(
            np.array([[0], [1]], dtype=np.int64),
            max_new_tokens=1,
            strategy="greedy",
            use_cache=False,
        )


@pytest.mark.parametrize(
    "bad_row",
    [
        [0.0, np.nan, -1.0],
        [0.0, np.inf, -1.0],
        [-np.inf, -np.inf, -np.inf],
    ],
)
def test_beam_generation_rejects_invalid_logits_before_ranking(bad_row):
    model = _tiny_model()
    _install_logits(model, np.array(bad_row))

    with pytest.raises(ValueError, match=r"NaN|\+inf|finite"):
        model.generate_beam(
            np.array([[0]], dtype=np.int64),
            max_new_tokens=1,
            beam_width=1,
        )


def test_beam_generation_still_allows_negative_infinity_candidates():
    model = _tiny_model()
    _install_logits(model, np.array([0.0, -np.inf, -1.0]))

    generated = model.generate_beam(
        np.array([[2]], dtype=np.int64),
        max_new_tokens=1,
        beam_width=1,
    )
    np.testing.assert_array_equal(generated, np.array([[2, 0]], dtype=np.int64))
