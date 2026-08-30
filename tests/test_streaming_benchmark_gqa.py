"""GQA/MQA coverage for the streaming benchmark."""

import argparse
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benchmark
import streaming_benchmark


def _args(**overrides):
    values = {
        "vocab": 16,
        "ctx": 4,
        "d": 8,
        "heads": 4,
        "kv_heads": 2,
        "layers": 1,
        "generate": 2,
        "warmup": 0,
        "repeats": 1,
        "seed": 19,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_gqa_streaming_report_exposes_compact_cache_architecture():
    report = streaming_benchmark.run_streaming_benchmark(_args())

    assert report["streaming_benchmark_schema"] == 1
    assert report["heads"] == 4
    assert report["kv_heads"] == 2
    assert report["kv_cache_head_ratio"] == 0.5
    assert report["kv_cache_head_reduction"] == 0.5
    assert report["kv_cache_bytes_per_token_batch"] == 64
    assert report["comparison"]["inside_window"]["outputs_match"] is True
    assert report["comparison"]["saturated_window"]["outputs_match"] is True
    json.dumps(report, allow_nan=False)


def test_mqa_streaming_reduces_parameters_and_cache_bytes():
    mha = streaming_benchmark.run_streaming_benchmark(_args(kv_heads=None))
    mqa = streaming_benchmark.run_streaming_benchmark(_args(kv_heads=1))

    assert mha["kv_heads"] == 4
    assert mha["kv_cache_bytes_per_token_batch"] == 128
    assert mqa["kv_heads"] == 1
    assert mqa["kv_cache_bytes_per_token_batch"] == 32
    assert mqa["parameters"] < mha["parameters"]


def test_general_and_streaming_benchmarks_use_same_cache_byte_formula():
    general_args = argparse.Namespace(
        vocab=16,
        ctx=4,
        d=8,
        heads=4,
        kv_heads=2,
        layers=1,
        batch=1,
        arch="llama",
        steps=1,
        generate=2,
        warmup=0,
        repeats=1,
        seed=23,
    )
    streaming_args = _args(seed=23)

    general = benchmark.run_benchmark(general_args)
    streaming = streaming_benchmark.run_streaming_benchmark(streaming_args)

    assert general["kv_cache_bytes_per_token_batch"] == 64
    assert streaming["kv_cache_bytes_per_token_batch"] == 64
    assert general["kv_cache_head_ratio"] == streaming["kv_cache_head_ratio"] == 0.5


def test_programmatic_call_without_kv_heads_remains_legacy_mha():
    args = _args()
    del args.kv_heads

    report = streaming_benchmark.run_streaming_benchmark(args)

    assert report["kv_heads"] == report["heads"] == 4
    assert report["kv_cache_head_ratio"] == 1.0
    assert report["kv_cache_head_reduction"] == 0.0


def test_numpy_integer_kv_heads_is_normalized_to_python_int():
    report = streaming_benchmark.run_streaming_benchmark(
        _args(kv_heads=np.int64(2))
    )
    assert type(report["kv_heads"]) is int
    assert report["kv_heads"] == 2


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, 0, -1])
def test_invalid_kv_head_type_or_range_is_rejected(value):
    error = TypeError if isinstance(value, (bool, np.bool_, float)) else ValueError
    with pytest.raises(error):
        streaming_benchmark._validate_args(_args(kv_heads=value))


def test_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError, match="--kv-heads must divide --heads"):
        streaming_benchmark._validate_args(_args(kv_heads=3))
    with pytest.raises(ValueError, match="--kv-heads must divide --heads"):
        streaming_benchmark._validate_args(_args(kv_heads=8))


def test_cli_json_accepts_kv_heads(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streaming_benchmark",
            "--vocab", "16",
            "--ctx", "4",
            "--d", "8",
            "--heads", "4",
            "--kv-heads", "2",
            "--layers", "1",
            "--generate", "2",
            "--warmup", "0",
            "--repeats", "1",
            "--json",
        ],
    )

    streaming_benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["heads"] == 4
    assert report["kv_heads"] == 2
    assert report["kv_cache_bytes_per_token_batch"] == 64
