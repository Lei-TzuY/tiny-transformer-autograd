"""Regression tests for strict generation-option validation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.transformer import GPT, _sample


def _model():
    return GPT(
        vocab_size=8,
        context_len=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    )


def _prompt():
    return np.array([[1, 2]], dtype=np.int64)


def _rng_state():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2], state[3], state[4]


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize(
    "overrides,error_type,match",
    [
        ({"max_new_tokens": True}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": np.bool_(False)}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": -1}, ValueError, "max_new_tokens"),
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": np.nan}, ValueError, "temperature"),
        ({"temperature": np.inf}, ValueError, "temperature"),
        ({"temperature": -np.inf}, ValueError, "temperature"),
        ({"temperature": 0.0}, ValueError, "temperature"),
        ({"top_k": True}, TypeError, "top_k"),
        ({"top_k": np.nan}, TypeError, "top_k"),
        ({"top_k": 1.5}, TypeError, "top_k"),
        ({"top_k": 0}, ValueError, "top_k"),
        ({"top_p": True}, TypeError, "top_p"),
        ({"top_p": np.nan}, ValueError, "top_p"),
        ({"top_p": np.inf}, ValueError, "top_p"),
        ({"top_p": 0.0}, ValueError, "top_p"),
        ({"top_p": 1.1}, ValueError, "top_p"),
        ({"beam_width": True}, TypeError, "beam_width"),
        ({"beam_width": 1.5}, TypeError, "beam_width"),
        ({"beam_width": 0}, ValueError, "beam_width"),
        ({"use_cache": 1}, TypeError, "use_cache"),
        ({"use_cache": "yes"}, TypeError, "use_cache"),
        ({"strategy": []}, TypeError, "strategy"),
        ({"strategy": "unknown"}, ValueError, "strategy"),
        # Options are validated even when the selected strategy would not use
        # them, rather than silently accepting a malformed public argument.
        ({"strategy": "greedy", "top_k": 0}, ValueError, "top_k"),
        ({"strategy": "beam", "top_p": np.nan}, ValueError, "top_p"),
        ({"strategy": "sample", "beam_width": 0}, ValueError, "beam_width"),
        ({"strategy": "beam", "use_cache": 1}, TypeError, "use_cache"),
    ],
)
def test_generate_rejects_invalid_options_before_inference_or_rng(
    monkeypatch, overrides, error_type, match
):
    model = _model()

    def forbidden_infer(*args, **kwargs):
        pytest.fail("generation reached inference before validating its options")

    monkeypatch.setattr(model, "infer", forbidden_infer)
    kwargs = {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "strategy": "sample",
        "beam_width": 2,
        "use_cache": True,
    }
    kwargs.update(overrides)

    np.random.seed(1234)
    before = _rng_state()
    with pytest.raises(error_type, match=match):
        model.generate(_prompt(), **kwargs)
    after = _rng_state()

    _assert_rng_state_equal(before, after)


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"max_new_tokens": True}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": -1}, ValueError, "max_new_tokens"),
        ({"beam_width": True}, TypeError, "beam_width"),
        ({"beam_width": 1.5}, TypeError, "beam_width"),
        ({"beam_width": 0}, ValueError, "beam_width"),
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": np.nan}, ValueError, "temperature"),
        ({"temperature": np.inf}, ValueError, "temperature"),
        ({"temperature": 0.0}, ValueError, "temperature"),
    ],
)
def test_generate_beam_validates_direct_call_before_inference(
    monkeypatch, kwargs, error_type, match
):
    model = _model()

    def forbidden_infer(*args, **kwargs):
        pytest.fail("beam search reached inference before validating its options")

    monkeypatch.setattr(model, "infer", forbidden_infer)
    options = {"max_new_tokens": 1, "beam_width": 2, "temperature": 1.0}
    options.update(kwargs)

    with pytest.raises(error_type, match=match):
        model.generate_beam(_prompt(), **options)


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": np.nan}, ValueError, "temperature"),
        ({"temperature": np.inf}, ValueError, "temperature"),
        ({"temperature": 0.0}, ValueError, "temperature"),
        ({"top_k": True}, TypeError, "top_k"),
        ({"top_k": np.nan}, TypeError, "top_k"),
        ({"top_k": 1.5}, TypeError, "top_k"),
        ({"top_k": 0}, ValueError, "top_k"),
        ({"top_p": True}, TypeError, "top_p"),
        ({"top_p": np.nan}, ValueError, "top_p"),
        ({"top_p": np.inf}, ValueError, "top_p"),
        ({"top_p": 0.0}, ValueError, "top_p"),
        ({"top_p": 1.1}, ValueError, "top_p"),
    ],
)
def test_sample_rejects_invalid_options_without_consuming_rng(
    kwargs, error_type, match
):
    np.random.seed(9876)
    before = _rng_state()
    with pytest.raises(error_type, match=match):
        _sample(np.array([10.0, 1.0, 0.0]), **kwargs)
    after = _rng_state()

    _assert_rng_state_equal(before, after)


def test_generation_accepts_valid_numpy_scalar_options():
    model = _model()
    prompt = _prompt()

    generated = model.generate(
        prompt,
        max_new_tokens=np.int64(0),
        temperature=np.float32(0.75),
        top_k=np.int64(2),
        top_p=np.float32(0.9),
        strategy="greedy",
        beam_width=np.int64(2),
        use_cache=np.bool_(False),
    )
    np.testing.assert_array_equal(generated, prompt)
    assert not np.shares_memory(generated, prompt)

    beamed = model.generate_beam(
        prompt,
        max_new_tokens=np.int64(0),
        beam_width=np.int64(2),
        temperature=np.float32(1.0),
    )
    np.testing.assert_array_equal(beamed, prompt)
    assert not np.shares_memory(beamed, prompt)

    assert _sample(
        np.array([10.0, 1.0, 0.0]),
        temperature=np.float32(1.0),
        top_k=np.int64(1),
        top_p=np.float32(1.0),
    ) == 0
