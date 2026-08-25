"""Tests for the strict-window versus streaming benchmark protocol."""

import argparse
import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streaming_benchmark import _validate_args, run_streaming_benchmark


def benchmark_args(**overrides):
    values = {
        "vocab": 16,
        "ctx": 4,
        "d": 8,
        "heads": 2,
        "layers": 2,
        "generate": 2,
        "warmup": 0,
        "repeats": 2,
        "seed": 7,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_report_separates_exact_and_different_semantic_regimes():
    report = run_streaming_benchmark(benchmark_args())

    assert report["streaming_benchmark_schema"] == 1
    assert report["arch"] == "llama"
    assert report["batch"] == 1
    assert report["repeats"] == 2
    assert report["seed"] == 7
    assert report["dtype"] == "float64"
    assert report["parameters"] > 0
    assert "python" in report["environment"]
    assert "numpy" in report["environment"]

    comparison = report["comparison"]
    assert comparison["strict_mode"] == "strict_window_refill"
    assert comparison["streaming_mode"] == "shifted_rope_cache"
    assert comparison["inside_window"] == {
        "prompt_length": 2,
        "generate_tokens": 2,
        "semantics_match": True,
        "outputs_match": True,
    }
    assert comparison["saturated_window"]["prompt_length"] == 4
    assert comparison["saturated_window"]["generate_tokens"] == 2
    assert comparison["saturated_window"]["semantics_match"] is False

    for name, samples in report["samples"].items():
        assert len(samples) == 2, name
        assert all(value > 0 for value in samples), name
        assert report["summary"][name]["n"] == 2
        assert report["summary"][name]["sample_stdev"] >= 0


def test_one_layer_marks_saturated_streaming_as_exact():
    report = run_streaming_benchmark(benchmark_args(layers=1, repeats=1))

    saturated = report["comparison"]["saturated_window"]
    assert saturated["semantics_match"] is True
    assert saturated["outputs_match"] is True


def test_speedup_samples_pair_matching_strict_and_streaming_measurements():
    durations = [4.0, 2.0, 1.0, 3.0, 8.0, 4.0, 2.0, 6.0]
    with mock.patch("streaming_benchmark._time_call", side_effect=durations):
        report = run_streaming_benchmark(benchmark_args())

    assert report["samples"]["inside_strict_seconds"] == [4.0, 3.0]
    assert report["samples"]["inside_streaming_seconds"] == [2.0, 1.0]
    assert report["samples"]["inside_streaming_speedup"] == [2.0, 3.0]
    assert report["inside_streaming_speedup"] == 2.5

    assert report["samples"]["saturated_strict_seconds"] == [8.0, 6.0]
    assert report["samples"]["saturated_streaming_seconds"] == [4.0, 2.0]
    assert report["samples"]["saturated_streaming_speedup"] == [2.0, 3.0]
    assert report["saturated_streaming_speedup"] == 2.5


def test_validation_rejects_invalid_dimensions_and_workload():
    with pytest.raises(ValueError, match="--vocab must be at least 2"):
        _validate_args(benchmark_args(vocab=1))
    with pytest.raises(ValueError, match="--ctx must be at least 2"):
        _validate_args(benchmark_args(ctx=1))
    with pytest.raises(ValueError, match="--generate must be at least 2"):
        _validate_args(benchmark_args(generate=1))
    with pytest.raises(ValueError, match="--d must be divisible"):
        _validate_args(benchmark_args(d=10, heads=4))
    with pytest.raises(ValueError, match="even head dimension"):
        _validate_args(benchmark_args(d=6, heads=2))


def test_validation_rejects_boolean_and_fractional_integer_fields():
    for field in (
        "vocab",
        "ctx",
        "d",
        "heads",
        "layers",
        "generate",
        "warmup",
        "repeats",
        "seed",
    ):
        for value in (True, 1.5):
            with pytest.raises(TypeError):
                _validate_args(benchmark_args(**{field: value}))


def test_validation_rejects_seed_outside_numpy_range():
    with pytest.raises(ValueError, match="--seed must be non-negative"):
        _validate_args(benchmark_args(seed=-1))
    with pytest.raises(ValueError, match="--seed must be at most"):
        _validate_args(benchmark_args(seed=2**32))


def test_programmatic_call_preserves_protocol_defaults():
    args = benchmark_args(repeats=1)
    del args.warmup
    del args.repeats

    report = run_streaming_benchmark(args)

    assert report["warmup"] == 1
    assert report["repeats"] == 1
    assert all(len(samples) == 1 for samples in report["samples"].values())


def test_benchmark_restores_caller_rng_after_success():
    np.random.seed(123456)
    before = np.random.get_state()

    run_streaming_benchmark(benchmark_args(repeats=1))

    assert _rng_equal(np.random.get_state(), before)


def test_benchmark_restores_caller_rng_after_timing_failure():
    np.random.seed(654321)
    before = np.random.get_state()

    with mock.patch(
        "streaming_benchmark._time_call",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            run_streaming_benchmark(benchmark_args(repeats=1))

    assert _rng_equal(np.random.get_state(), before)
