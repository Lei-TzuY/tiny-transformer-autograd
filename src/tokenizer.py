"""Character and educational byte-pair tokenizers."""

import re
from collections import Counter

import numpy as np


class CharTokenizer:
    """Map each distinct character to one token."""

    kind = "char"

    def __init__(self, vocab):
        self.vocab = _normalise_vocab(vocab, character_tokens=True)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    @classmethod
    def train(cls, text):
        _validate_text(text)
        if not text:
            raise ValueError("cannot train a character tokenizer on empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        _validate_text(text)
        return np.array([self.stoi[token] for token in text], dtype=np.int64)

    def decode(self, ids):
        ids = _validate_decode_ids(ids, self.vocab_size)
        return "".join(self.vocab[int(i)] for i in ids)

    def state_dict(self):
        return {"kind": self.kind, "vocab": self.vocab.copy()}


class BPETokenizer:
    """Learn BPE merges inside words and whitespace runs."""

    kind = "bpe"

    def __init__(self, vocab, merges):
        self.vocab = _normalise_vocab(vocab)
        self.merges = _normalise_merges(merges, self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    @classmethod
    def train(cls, text, num_merges=100):
        _validate_text(text)
        if not text:
            raise ValueError("cannot train a BPE tokenizer on empty text")
        if (
            not isinstance(num_merges, (int, np.integer))
            or isinstance(num_merges, (bool, np.bool_))
            or num_merges < 0
        ):
            raise ValueError("num_merges must be a non-negative integer")

        segments = _segments(text)
        vocab = sorted(set(text))
        merges = []

        for _ in range(int(num_merges)):
            counts = Counter(
                pair
                for segment in segments
                for pair in zip(segment, segment[1:])
            )
            if not counts:
                break
            best = min(counts, key=lambda pair: (-counts[pair], pair))
            segments = [_merge(segment, best) for segment in segments]
            merges.append(best)
            token = "".join(best)
            if token not in vocab:
                vocab.append(token)

        return cls(vocab, merges)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        _validate_text(text)
        segments = _segments(text)
        for pair in self.merges:
            segments = [_merge(segment, pair) for segment in segments]
        return np.array(
            [self.stoi[token] for segment in segments for token in segment],
            dtype=np.int64,
        )

    def decode(self, ids):
        ids = _validate_decode_ids(ids, self.vocab_size)
        return "".join(self.vocab[int(i)] for i in ids)

    def state_dict(self):
        return {
            "kind": self.kind,
            "vocab": self.vocab.copy(),
            "merges": [list(pair) for pair in self.merges],
        }


def build_tokenizer(kind, text, bpe_merges=100):
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, num_merges=bpe_merges)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def tokenizer_from_state_dict(state):
    """Rebuild a tokenizer from validated, checkpoint-safe state."""
    if not isinstance(state, dict):
        raise TypeError("tokenizer state must be a dictionary")
    if "kind" not in state:
        raise ValueError("tokenizer state is missing 'kind'")
    kind = state["kind"]
    if kind == "char":
        if "vocab" not in state:
            raise ValueError("character tokenizer state is missing 'vocab'")
        return CharTokenizer(state["vocab"])
    if kind == "bpe":
        missing = [key for key in ("vocab", "merges") if key not in state]
        if missing:
            raise ValueError(
                "BPE tokenizer state is missing " + ", ".join(repr(key) for key in missing)
            )
        return BPETokenizer(state["vocab"], state["merges"])
    raise ValueError(f"unknown tokenizer kind: {kind}")


def _normalise_vocab(vocab, *, character_tokens=False):
    if isinstance(vocab, str):
        tokens = list(vocab)
    else:
        try:
            tokens = list(vocab)
        except TypeError as exc:
            raise TypeError("tokenizer vocab must be an iterable of strings") from exc
    if not tokens:
        raise ValueError("tokenizer vocab must not be empty")

    seen = set()
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise TypeError(f"tokenizer vocab entry {index} must be a string")
        if not token:
            raise ValueError(f"tokenizer vocab entry {index} must not be empty")
        if character_tokens and len(token) != 1:
            raise ValueError(
                f"character tokenizer vocab entry {index} must contain one character"
            )
        if token in seen:
            raise ValueError(f"tokenizer vocab contains duplicate token {token!r}")
        seen.add(token)
    return tokens


def _normalise_merges(merges, vocab):
    try:
        raw_merges = list(merges)
    except TypeError as exc:
        raise TypeError("BPE merges must be an iterable of token pairs") from exc

    pairs = []
    for index, pair in enumerate(raw_merges):
        if isinstance(pair, str):
            raise TypeError(f"BPE merge {index} must be a pair, not a string")
        try:
            pair = tuple(pair)
        except TypeError as exc:
            raise TypeError(f"BPE merge {index} must be an iterable pair") from exc
        if len(pair) != 2:
            raise ValueError(f"BPE merge {index} must contain exactly two tokens")
        if not all(isinstance(token, str) for token in pair):
            raise TypeError(f"BPE merge {index} tokens must be strings")
        if not all(pair):
            raise ValueError(f"BPE merge {index} tokens must not be empty")
        pairs.append(pair)

    vocab_set = set(vocab)
    outputs = {left + right for left, right in pairs}
    available = vocab_set - outputs
    for index, (left, right) in enumerate(pairs):
        if left not in available or right not in available:
            raise ValueError(
                f"BPE merge {index} references a token unavailable at that merge step"
            )
        merged = left + right
        if merged not in vocab_set:
            raise ValueError(
                f"BPE merge {index} output {merged!r} is missing from the vocab"
            )
        available.add(merged)
    return pairs


def _validate_decode_ids(ids, vocab_size):
    values = np.asarray(ids)
    if values.ndim != 1:
        raise ValueError("token ids to decode must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(values.dtype, np.integer) or values.dtype == np.bool_:
        raise TypeError("token ids to decode must be integers")
    if np.any(values < 0) or np.any(values >= vocab_size):
        raise ValueError(f"token ids to decode must be in [0, {vocab_size})")
    return values


def _validate_text(text):
    if not isinstance(text, str):
        raise TypeError("tokenizer text must be a string")


def _segments(text):
    return [list(segment) for segment in re.findall(r"\s+|[^\s]+", text)]


def _merge(tokens, pair):
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged
