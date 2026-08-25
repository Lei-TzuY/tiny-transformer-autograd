"""Regression tests for decoding empty token sequences."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import BPETokenizer, CharTokenizer


@pytest.fixture(params=["char", "bpe"])
def tokenizer(request):
    if request.param == "char":
        return CharTokenizer("abc")
    return BPETokenizer(["a", "b", "c", "ab"], [("a", "b")])


@pytest.mark.parametrize(
    "ids",
    [
        [],
        (),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    ],
)
def test_empty_one_dimensional_sequences_decode_to_empty_string(tokenizer, ids):
    assert tokenizer.decode(ids) == ""


@pytest.mark.parametrize(
    "ids",
    [
        [0.0],
        [True],
        np.array([0.0], dtype=np.float64),
        np.array([True], dtype=np.bool_),
    ],
)
def test_nonempty_noninteger_sequences_remain_rejected(tokenizer, ids):
    with pytest.raises(TypeError, match="token ids to decode must be integers"):
        tokenizer.decode(ids)


def test_empty_multidimensional_array_remains_rejected(tokenizer):
    ids = np.empty((1, 0), dtype=np.int64)

    with pytest.raises(ValueError, match="one-dimensional"):
        tokenizer.decode(ids)
