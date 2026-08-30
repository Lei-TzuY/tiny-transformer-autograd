"""GQA/MQA coverage for the general benchmark entry point."""

import argparse
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benchmark


def _args(**overrides):
    values = {
        "vocab": 16,
        "ctx": 4,
        "d": 8,
        "heads": 4,
        "kv_heads": 2,
        "layers": 1,
        "batch": 1,
        "arch": "gpt",
        "steps": 1,
        "generate": 1,
        "warmup": 0,
        "repeats": 1,
        "seed": 17,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_gqa_report_exposes_compact_cache_architecture():
    report = benchmark.run_benchmark(_args())

    assert report["benchmark_schema"] == 1
    assert report["heads"] == 4
    assert report["kv_heads"] == 2
    assert report["kv_cache_head_ratio"] == 0.5
    assert report["kv_cache_head_reduction"] == 0.5
    assert report["kv_cache_bytes_per_token_batch"] == 64
    json.dumps(report, allow_nan=False)


def test_mqa_reduces_parameters_and_cache_bytes_from_mha():
    mha = benchmark.run_benchmark(_args(kv_heads=None))
    mqa = benchmark.run_benchmark(_args(kv_heads=1))

    assert mha["kv_heads"] == mha["heads"] == 4
    assert mha["kv_cache_head_ratio"] == 1.0
    assert mha["kv_cache_head_reduction"] == 0.0
    assert mha["kv_cache_bytes_per_token_batch"] == 128

    assert mqa["kv_heads"] == 1
    assert mqa["kv_cache_head_ratio"] == 0.25
    assert mqa["kv_cache_head_reduction"] == 0.75
    assert mqa["kv_cache_bytes_per_token_batch"] == 32
    assert mqa["parameters"] < mha["parameters"]


def test_programmatic_call_without_kv_heads_remains_legacy_mha():
    args = _args()
    del args.kv_heads

    report = benchmark.run_benchmark(args)

    assert report["kv_heads"] == report["heads"] == 4
    assert report["kv_cache_head_ratio"] == 1.0
    assert report["kv_cache_head_reduction"] == 0.0


def test_numpy_integer_kv_heads_is_normalized_to_python_int():
    report = benchmark.run_benchmark(_args(kv_heads=np.int64(2)))
    assert type(report["kv_heads"]) is int
    assert report["kv_heads"] == 2


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, 0, -1])
def test_invalid_kv_head_type_or_range_is_rejected(value):
    error = TypeError if isinstance(value, (bool, np.bool_, float)) else ValueError
    with pytest.raises(error):
        benchmark._validate_args(_args(kv_heads=value))


def test_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError, match="--kv-heads must divide --heads"):
        benchmark._validate_args(_args(kv_heads=3))
    with pytest.raises(ValueError, match="--kv-heads must divide --heads"):
        benchmark._validate_args(_args(kv_heads=8))


def test_cli_json_accepts_kv_heads(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--vocab", "16",
            "--ctx", "4",
            "--d", "8",
            "--heads", "4",
            "--kv-heads", "2",
            "--layers", "1",
            "--batch", "1",
            "--steps", "1",
            "--generate", "1",
            "--warmup", "0",
            "--repeats", "1",
            "--json",
        ],
    )

    benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["heads"] == 4
    assert report["kv_heads"] == 2
    assert report["kv_cache_bytes_per_token_batch"] == 64
