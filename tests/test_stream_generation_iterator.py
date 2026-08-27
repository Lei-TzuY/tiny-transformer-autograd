"""Incremental token-yielding regressions for shifted-cache generation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import stream_generate_iter as exported_stream_generate_iter
from nn.streaming import stream_generate, stream_generate_iter
from nn.transformer import GPT


def _model(pos_encoding="rope"):
    np.random.seed(71)
    return GPT(
        vocab_size=19,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        pos_encoding=pos_encoding,
    )


def _generated_suffix(iterator):
    steps = list(iterator)
    if not steps:
        return np.empty((1, 0), dtype=np.int64)
    return np.stack(steps, axis=1)


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_public_nn_export_is_the_streaming_iterator():
    assert exported_stream_generate_iter is stream_generate_iter


def test_fully_consumed_greedy_iterator_matches_stream_generate_past_window():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    model = _model()

    full = stream_generate(model, prompt, 8, strategy="greedy")
    suffix = _generated_suffix(
        stream_generate_iter(model, prompt, 8, strategy="greedy")
    )

    np.testing.assert_array_equal(suffix, full[:, prompt.shape[1]:])


def test_fully_consumed_sample_iterator_matches_values_and_rng_state():
    prompt = np.array([[1, 3, 5]], dtype=np.int64)
    model = _model()

    np.random.seed(1234)
    full = stream_generate(
        model,
        prompt,
        7,
        strategy="sample",
        temperature=0.7,
        top_k=5,
        top_p=0.8,
    )
    full_rng = np.random.get_state()

    np.random.seed(1234)
    suffix = _generated_suffix(
        stream_generate_iter(
            model,
            prompt,
            7,
            strategy="sample",
            temperature=0.7,
            top_k=5,
            top_p=0.8,
        )
    )
    iter_rng = np.random.get_state()

    np.testing.assert_array_equal(suffix, full[:, prompt.shape[1]:])
    _assert_rng_state_equal(iter_rng, full_rng)


def test_masked_batched_iterator_matches_stream_generate():
    prompt = np.array(
        [
            [0, 1, 3, 5],
            [0, 0, 2, 4],
        ],
        dtype=np.int64,
    )
    mask = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    model = _model()

    full = stream_generate(
        model,
        prompt,
        6,
        strategy="greedy",
        attention_mask=mask,
    )
    steps = list(
        stream_generate_iter(
            model,
            prompt,
            6,
            strategy="greedy",
            attention_mask=mask,
        )
    )

    assert all(step.shape == (2,) and step.dtype == np.int64 for step in steps)
    np.testing.assert_array_equal(
        np.stack(steps, axis=1),
        full[:, prompt.shape[1]:],
    )


def test_iterator_creation_is_inference_lazy_and_one_next_advances_one_step():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    model = _model()
    widths = []
    infer = model.infer

    def counted(idx, *args, **kwargs):
        widths.append(np.asarray(idx).shape[1])
        return infer(idx, *args, **kwargs)

    model.infer = counted
    iterator = stream_generate_iter(model, prompt, 5, strategy="greedy")
    assert widths == []

    first = next(iterator)
    assert first.shape == (1,)
    assert widths == [4]

    second = next(iterator)
    assert second.shape == (1,)
    assert widths == [4, 1]

    iterator.close()
    assert widths == [4, 1]


def test_mutating_yielded_token_cannot_change_later_decoding():
    prompt = np.array([[1, 3, 5, 7]], dtype=np.int64)
    model = _model()
    baseline = list(stream_generate_iter(model, prompt, 5, strategy="greedy"))

    iterator = stream_generate_iter(model, prompt, 5, strategy="greedy")
    first = next(iterator)
    first[...] = (first + 1) % model.vocab_size
    remaining = list(iterator)

    assert len(remaining) == len(baseline) - 1
    for actual, expected in zip(remaining, baseline[1:]):
        np.testing.assert_array_equal(actual, expected)


def test_zero_token_iterator_is_empty_without_inference():
    prompt = np.array([[1, 3]], dtype=np.int64)
    model = _model()

    def fail_infer(*_args, **_kwargs):
        raise AssertionError("zero-token iterator must not run inference")

    model.infer = fail_infer
    iterator = stream_generate_iter(model, prompt, 0, strategy="greedy")

    assert list(iterator) == []


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: stream_generate_iter(
                _model(pos_encoding="learned"),
                np.array([[1, 2]]),
                1,
            ),
            ValueError,
            "pos_encoding='rope'",
        ),
        (
            lambda: stream_generate_iter(
                _model(),
                np.array([[1, 2]]),
                1,
                strategy="beam",
            ),
            ValueError,
            "sample.*greedy",
        ),
        (
            lambda: stream_generate_iter(
                _model(),
                np.array([[0, 1]]),
                1,
                attention_mask=np.array([[1, 0]]),
            ),
            ValueError,
            "left-padded",
        ),
    ],
)
def test_iterator_validation_is_eager(call, error, message):
    with pytest.raises(error, match=message):
        call()
