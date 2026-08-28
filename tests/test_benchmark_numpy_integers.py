"""Regression coverage for NumPy integer benchmark arguments."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import run_benchmark


def test_narrow_numpy_integer_arguments_are_normalized_before_arithmetic():
    args = argparse.Namespace(
        vocab=np.int8(16),
        ctx=np.int8(4),
        d=np.int8(32),
        heads=np.int8(4),
        layers=np.int8(1),
        batch=np.int8(1),
        arch="gpt",
        steps=np.int8(1),
        generate=np.int8(1),
        warmup=np.int8(0),
        repeats=np.int8(1),
        seed=np.int8(7),
    )

    # Before normalization, 4 * np.int8(32) overflows to -128. Make that
    # latent NumPy-scalar arithmetic bug deterministic instead of warning-only.
    with np.errstate(over="raise"):
        report = run_benchmark(args)

    assert report["vocab"] == 16
    assert report["context_len"] == 4
    assert report["d_model"] == 32
    assert report["heads"] == 4
    assert report["layers"] == 1
    assert report["batch"] == 1
    assert report["d_ff"] == 128
    assert report["steps"] == 1
    assert report["generate_tokens"] == 1
    assert report["warmup"] == 0
    assert report["repeats"] == 1
    assert report["seed"] == 7

    # Normalization is local to the benchmark and must not rewrite caller state.
    assert isinstance(args.d, np.int8)
    assert isinstance(args.ctx, np.int8)
    assert isinstance(args.batch, np.int8)
    assert isinstance(args.steps, np.int8)
    assert isinstance(args.generate, np.int8)
