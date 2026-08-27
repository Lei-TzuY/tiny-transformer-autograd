"""Regression coverage for paired cached/uncached benchmark ordering."""

import argparse
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benchmark


def _args():
    return argparse.Namespace(
        vocab=32,
        ctx=4,
        d=8,
        heads=2,
        layers=1,
        batch=1,
        arch="gpt",
        steps=1,
        generate=2,
        warmup=0,
        repeats=4,
        seed=7,
    )


def test_generation_measurements_alternate_order_but_preserve_pairing():
    calls = []
    durations = {
        "forward_no_grad_once": 1.0,
        "infer_once": 0.5,
        "generate_cached_once": 0.25,
        "generate_uncached_once": 1.0,
    }

    def fake_time_call(function):
        calls.append(function.__name__)
        return durations[function.__name__]

    with mock.patch("benchmark._time_call", side_effect=fake_time_call):
        report = benchmark.run_benchmark(_args())

    assert calls[-8:] == [
        "generate_cached_once",
        "generate_uncached_once",
        "generate_uncached_once",
        "generate_cached_once",
        "generate_cached_once",
        "generate_uncached_once",
        "generate_uncached_once",
        "generate_cached_once",
    ]
    assert report["samples"]["generate_cached_seconds"] == [0.25] * 4
    assert report["samples"]["generate_uncached_seconds"] == [1.0] * 4
    assert report["samples"]["cache_speedup"] == [4.0] * 4
