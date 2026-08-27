"""Validation tests for tokenizer construction and serialized state."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_state_dict


@pytest.mark.parametrize(
    ("tokenizer", "text"),
    [
        (CharTokenizer.train("ab βa"), "βaba "),
        (BPETokenizer.train("banana bandana banana", num_merges=8), "banana bandana"),
    ],
    ids=["char", "bpe"],
)
def test_valid_state_roundtrip_preserves_encoding_and_decoding(tokenizer, text):
    restored = tokenizer_from_state_dict(tokenizer.state_dict())

    expected = tokenizer.encode(text)
    actual = restored.encode(text)

    np.testing.assert_array_equal(actual, expected)
    assert restored.decode(actual) == tokenizer.decode(expected)
    assert restored.state_dict() == tokenizer.state_dict()


def test_state_dict_does_not_alias_the_live_vocab():
    tokenizer = CharTokenizer.train("abc")
    state = tokenizer.state_dict()

    state["vocab"][0] = "z"

    assert tokenizer.vocab == ["a", "b", "c"]
    assert tokenizer.encode("abc").tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("state", "error", "message"),
    [
        (None, TypeError, "must be a dictionary"),
        ({}, ValueError, "missing 'kind'"),
        ({"kind": "char"}, ValueError, "missing 'vocab'"),
        ({"kind": "bpe", "vocab": ["a"]}, ValueError, "missing 'merges'"),
        ({"kind": "word", "vocab": ["a"]}, ValueError, "unknown tokenizer kind"),
        (
            {"kind": "char", "vocab": ["a", "a"]},
            ValueError,
            "duplicate token",
        ),
        (
            {"kind": "char", "vocab": ["ab"]},
            ValueError,
            "one character",
        ),
        (
            {"kind": "bpe", "vocab": ["a", "b"], "merges": [["a", "b"]]},
            ValueError,
            "output 'ab' is missing",
        ),
        (
            {
                "kind": "bpe",
                "vocab": ["a", "b", "c", "ab", "abc"],
                "merges": [["ab", "c"], ["a", "b"]],
            },
            ValueError,
            "unavailable at that merge step",
        ),
    ],
)
def test_rejects_malformed_serialized_state(state, error, message):
    with pytest.raises(error, match=message):
        tokenizer_from_state_dict(state)


@pytest.mark.parametrize(
    ("ids", "error", "message"),
    [
        ([0.0, 1.0], TypeError, "must be integers"),
        ([True, False], TypeError, "must be integers"),
        ([[0, 1]], ValueError, "one-dimensional"),
        ([-1], ValueError, r"\[0, 3\)"),
        ([3], ValueError, r"\[0, 3\)"),
    ],
)
def test_decode_rejects_invalid_token_ids(ids, error, message):
    tokenizer = CharTokenizer("abc")

    with pytest.raises(error, match=message):
        tokenizer.decode(ids)


@pytest.mark.parametrize("tokenizer_cls", [CharTokenizer, BPETokenizer])
def test_encode_requires_text(tokenizer_cls):
    tokenizer = (
        tokenizer_cls("ab")
        if tokenizer_cls is CharTokenizer
        else tokenizer_cls(["a", "b"], [])
    )

    with pytest.raises(TypeError, match="text must be a string"):
        tokenizer.encode(123)


@pytest.mark.parametrize(
    "tokenizer",
    [
        CharTokenizer("abc"),
        BPETokenizer(["a", "b", " ", "ab"], [("a", "b")]),
    ],
    ids=["char", "bpe"],
)
def test_encode_rejects_characters_absent_from_vocabulary(tokenizer):
    with pytest.raises(
        ValueError,
        match=r"not present in the tokenizer vocabulary: 'x', 'z'",
    ):
        tokenizer.encode("zax")


def test_bpe_encode_validates_raw_characters_before_merging():
    tokenizer = BPETokenizer(["a", "b", "ab"], [("a", "b")])

    with pytest.raises(
        ValueError,
        match=r"not present in the tokenizer vocabulary: 'c'",
    ):
        tokenizer.encode("abc")


def test_empty_encode_remains_supported_with_int64_result():
    tokenizers = [
        CharTokenizer("abc"),
        BPETokenizer(["a", "b", "ab"], [("a", "b")]),
    ]

    for tokenizer in tokenizers:
        encoded = tokenizer.encode("")
        assert encoded.shape == (0,)
        assert encoded.dtype == np.int64


@pytest.mark.parametrize(
    "call",
    [
        lambda: CharTokenizer.train(""),
        lambda: BPETokenizer.train("", num_merges=1),
    ],
)
def test_training_rejects_empty_text(call):
    with pytest.raises(ValueError, match="empty text"):
        call()


@pytest.mark.parametrize("num_merges", [-1, np.int64(-2)])
def test_bpe_training_rejects_negative_merge_count(num_merges):
    with pytest.raises(ValueError, match="non-negative integer"):
        BPETokenizer.train("ababa", num_merges=num_merges)


@pytest.mark.parametrize("num_merges", [1.5, True, np.bool_(False), "1", None])
def test_bpe_training_rejects_non_integer_merge_count(num_merges):
    with pytest.raises(TypeError, match="non-negative integer"):
        BPETokenizer.train("ababa", num_merges=num_merges)


def test_bpe_training_accepts_numpy_integer_merge_count():
    tokenizer = BPETokenizer.train("ababa", num_merges=np.int64(2))

    assert tokenizer.merges
    np.testing.assert_array_equal(
        tokenizer.encode("ababa"),
        BPETokenizer.train("ababa", num_merges=2).encode("ababa"),
    )
