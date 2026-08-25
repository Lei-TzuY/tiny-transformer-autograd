"""Validation parity for the public ``nn.beam_generate`` API."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import beam_generate
from nn.transformer import GPT


def _model():
    np.random.seed(79)
    return GPT(
        vocab_size=9,
        context_len=5,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_new_tokens": True}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": 1.0}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": np.float64(1.0)}, TypeError, "max_new_tokens"),
        ({"max_new_tokens": -1}, ValueError, "max_new_tokens"),
        ({"beam_width": True}, TypeError, "beam_width"),
        ({"beam_width": 2.0}, TypeError, "beam_width"),
        ({"beam_width": np.float64(2.0)}, TypeError, "beam_width"),
        ({"beam_width": 0}, ValueError, "beam_width"),
        ({"temperature": True}, TypeError, "temperature"),
        ({"temperature": np.nan}, ValueError, "finite"),
        ({"temperature": np.inf}, ValueError, "finite"),
        ({"temperature": -np.inf}, ValueError, "finite"),
        ({"temperature": 0.0}, ValueError, "positive"),
        ({"use_cache": 1}, TypeError, "use_cache"),
    ],
)
def test_invalid_options_fail_before_inference(kwargs, error, message):
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    def unexpected_infer(*args, **kwargs):
        raise AssertionError("invalid public options must fail before inference")

    model.infer = unexpected_infer
    options = {"max_new_tokens": 1}
    options.update(kwargs)

    with pytest.raises(error, match=message):
        beam_generate(model, prompt, **options)


def test_numpy_scalar_options_remain_supported():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    result = beam_generate(
        model,
        prompt,
        np.int64(2),
        beam_width=np.int64(2),
        temperature=np.float64(0.9),
        use_cache=np.bool_(True),
    )

    assert result.shape == (1, 4)


def test_zero_token_generation_validates_input_without_running_inference():
    model = _model()
    prompt = np.array([[1, 2]], dtype=np.int64)

    def unexpected_infer(*args, **kwargs):
        raise AssertionError("zero-token generation must not run model inference")

    model.infer = unexpected_infer
    result = beam_generate(
        model,
        prompt,
        np.int64(0),
        beam_width=np.int64(2),
        temperature=np.float64(1.0),
        use_cache=np.bool_(False),
    )

    np.testing.assert_array_equal(result, prompt)
    assert result is not prompt
