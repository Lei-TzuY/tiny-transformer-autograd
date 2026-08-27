"""Stop-token semantics for incremental shifted-cache generation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nn.streaming as streaming
from nn.streaming import stream_generate, stream_generate_iter
from nn.transformer import GPT


def _model(context_len=8):
    np.random.seed(83)
    return GPT(
        vocab_size=19,
        context_len=context_len,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        pos_encoding="rope",
    )


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _scripted_infer(model, selected_by_call, observed_inputs):
    calls = {"count": 0}

    def infer(idx, cache=None, **_kwargs):
        tokens = np.asarray(idx)
        observed_inputs.append(tokens.copy())
        call = calls["count"]
        calls["count"] += 1
        selected = np.asarray(selected_by_call[call], dtype=np.int64)
        batch, time = tokens.shape
        assert selected.shape == (batch,)

        logits = np.full((batch, time, model.vocab_size), -10.0)
        logits[np.arange(batch), -1, selected] = 10.0
        past = 0 if cache is None else cache[0]["k"].shape[2]
        cache_len = past + time
        key = np.zeros((batch, 1, cache_len, 2), dtype=np.float64)
        return logits, [{"k": key, "v": key.copy()}]

    return infer


def test_single_row_stop_includes_token_and_avoids_next_inference():
    prompt = np.array([[1, 3, 5]], dtype=np.int64)
    baseline_model = _model()
    first = stream_generate(baseline_model, prompt, 1, strategy="greedy")
    stop_token_id = int(first[0, -1])

    model = _model()
    widths = []
    infer = model.infer

    def counted(idx, *args, **kwargs):
        widths.append(np.asarray(idx).shape[1])
        return infer(idx, *args, **kwargs)

    model.infer = counted
    result = stream_generate(
        model,
        prompt,
        10,
        strategy="greedy",
        stop_token_id=stop_token_id,
    )

    assert result.shape == (1, prompt.shape[1] + 1)
    assert result[0, -1] == stop_token_id
    assert widths == [prompt.shape[1]]


def test_prompt_stop_token_does_not_finish_before_generation():
    prompt = np.array([[1, 3, 5]], dtype=np.int64)
    stop_token_id = int(prompt[0, -1])
    model = _model()
    widths = []
    infer = model.infer

    def counted(idx, *args, **kwargs):
        widths.append(np.asarray(idx).shape[1])
        return infer(idx, *args, **kwargs)

    model.infer = counted
    iterator = stream_generate_iter(
        model,
        prompt,
        1,
        strategy="greedy",
        stop_token_id=stop_token_id,
    )

    generated = list(iterator)
    assert len(generated) == 1
    assert widths == [prompt.shape[1]]


def test_batched_rows_enter_absorbing_stop_state_until_all_finish():
    stop_token_id = 2
    prompt = np.array([[1, 3], [4, 6]], dtype=np.int64)
    model = _model()
    observed_inputs = []
    model.infer = _scripted_infer(
        model,
        [
            [stop_token_id, 5],
            [7, stop_token_id],
        ],
        observed_inputs,
    )

    steps = list(
        stream_generate_iter(
            model,
            prompt,
            8,
            strategy="greedy",
            stop_token_id=stop_token_id,
        )
    )

    assert len(steps) == 2
    np.testing.assert_array_equal(steps[0], [stop_token_id, 5])
    np.testing.assert_array_equal(steps[1], [stop_token_id, stop_token_id])
    assert len(observed_inputs) == 2
    np.testing.assert_array_equal(
        observed_inputs[1],
        np.array([[stop_token_id], [5]], dtype=np.int64),
    )


def test_stream_generate_keeps_rectangular_absorbing_stop_columns():
    stop_token_id = 2
    prompt = np.array([[1, 3], [4, 6]], dtype=np.int64)
    model = _model()
    observed_inputs = []
    model.infer = _scripted_infer(
        model,
        [
            [stop_token_id, 5],
            [7, stop_token_id],
        ],
        observed_inputs,
    )

    result = stream_generate(
        model,
        prompt,
        8,
        strategy="greedy",
        stop_token_id=stop_token_id,
    )

    expected = np.array(
        [
            [1, 3, stop_token_id, stop_token_id],
            [4, 6, 5, stop_token_id],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(result, expected)
    assert len(observed_inputs) == 2


def test_sampling_skips_rng_draws_for_finished_rows(monkeypatch):
    stop_token_id = 2
    prompt = np.array([[1, 3], [4, 6]], dtype=np.int64)
    model = _model()
    observed_inputs = []
    model.infer = _scripted_infer(
        model,
        [
            [0, 0],
            [0, 0],
        ],
        observed_inputs,
    )
    sampled = iter([stop_token_id, 5, stop_token_id])
    calls = []

    def fake_sample(*_args, **_kwargs):
        calls.append(None)
        return next(sampled)

    monkeypatch.setattr(streaming, "_sample", fake_sample)
    steps = list(
        stream_generate_iter(
            model,
            prompt,
            8,
            strategy="sample",
            stop_token_id=stop_token_id,
        )
    )

    assert len(calls) == 3
    np.testing.assert_array_equal(steps[0], [stop_token_id, 5])
    np.testing.assert_array_equal(steps[1], [stop_token_id, stop_token_id])


def test_explicit_none_preserves_sampling_values_and_rng_state():
    prompt = np.array([[1, 3, 5]], dtype=np.int64)

    np.random.seed(2026)
    omitted = stream_generate(
        _model(),
        prompt,
        7,
        strategy="sample",
        temperature=0.8,
        top_k=6,
        top_p=0.9,
    )
    omitted_rng = np.random.get_state()

    np.random.seed(2026)
    explicit_none = stream_generate(
        _model(),
        prompt,
        7,
        strategy="sample",
        temperature=0.8,
        top_k=6,
        top_p=0.9,
        stop_token_id=None,
    )
    explicit_rng = np.random.get_state()

    np.testing.assert_array_equal(explicit_none, omitted)
    _assert_rng_state_equal(explicit_rng, omitted_rng)


@pytest.mark.parametrize(
    "bad_value",
    [True, np.bool_(False), 1.5, np.float64(2.0), "2"],
)
def test_stop_token_rejects_non_integer_types_eagerly(bad_value):
    model = _model()

    def fail_infer(*_args, **_kwargs):
        raise AssertionError("invalid stop_token_id must fail before inference")

    model.infer = fail_infer
    with pytest.raises(TypeError, match="stop_token_id.*integer"):
        stream_generate_iter(
            model,
            np.array([[1, 3]], dtype=np.int64),
            1,
            stop_token_id=bad_value,
        )


@pytest.mark.parametrize("bad_value", [-1, 19, np.int64(20)])
def test_stop_token_rejects_out_of_vocabulary_ids_eagerly(bad_value):
    model = _model()

    def fail_infer(*_args, **_kwargs):
        raise AssertionError("invalid stop_token_id must fail before inference")

    model.infer = fail_infer
    with pytest.raises(ValueError, match=r"stop_token_id.*\[0, 18\]"):
        stream_generate_iter(
            model,
            np.array([[1, 3]], dtype=np.int64),
            1,
            stop_token_id=bad_value,
        )


def test_numpy_integer_stop_token_is_accepted_without_inference_for_zero_budget():
    model = _model()

    def fail_infer(*_args, **_kwargs):
        raise AssertionError("zero-token generation must not run inference")

    model.infer = fail_infer
    iterator = stream_generate_iter(
        model,
        np.array([[1, 3]], dtype=np.int64),
        0,
        stop_token_id=np.int64(3),
    )

    assert list(iterator) == []
