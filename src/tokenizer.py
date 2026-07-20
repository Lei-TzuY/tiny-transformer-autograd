"""Character and educational byte-pair tokenizers."""

import re
from collections import Counter

import numpy as np


class CharTokenizer:
    """Map each distinct character to one token."""

    kind = "char"

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    @classmethod
    def train(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        return np.array([self.stoi[token] for token in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.vocab[int(i)] for i in ids)

    def state_dict(self):
        return {"kind": self.kind, "vocab": self.vocab}


class BPETokenizer:
    """Learn BPE merges inside words and whitespace runs."""

    kind = "bpe"

    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.merges = [tuple(pair) for pair in merges]

    @classmethod
    def train(cls, text, num_merges=100):
        segments = _segments(text)
        vocab = sorted(set(text))
        merges = []

        for _ in range(num_merges):
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
        segments = _segments(text)
        for pair in self.merges:
            segments = [_merge(segment, pair) for segment in segments]
        return np.array(
            [self.stoi[token] for segment in segments for token in segment],
            dtype=np.int64,
        )

    def decode(self, ids):
        return "".join(self.vocab[int(i)] for i in ids)

    def state_dict(self):
        return {
            "kind": self.kind,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }


def build_tokenizer(kind, text, bpe_merges=100):
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, num_merges=bpe_merges)
    raise ValueError(f"unknown tokenizer kind: {kind}")


def tokenizer_from_state_dict(state):
    if state["kind"] == "char":
        return CharTokenizer(state["vocab"])
    if state["kind"] == "bpe":
        return BPETokenizer(state["vocab"], state["merges"])
    raise ValueError(f"unknown tokenizer kind: {state['kind']}")


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
