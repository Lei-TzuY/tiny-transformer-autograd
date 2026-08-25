"""Tests for the reproducible benchmark protocol and report schema."""

import argparse
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import _summarize, _validate_args, environment_metadata, run_benchmark


def benchmark_args(**overrides):
    values = {
        "vocab": 32,
        "ctx": 4,
        "d": 8,
        "heads": 2,
        "layers": 1,
        "batch": 1,
        "arch": "gpt",
        "steps": 1,
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


class BenchmarkTest(unittest.TestCase):
    def test_report_contains_metadata_and_all_samples(self):
        report = run_benchmark(benchmark_args())

        self.assertEqual(report["benchmark_schema"], 1)
        self.assertEqual(report["repeats"], 2)
        self.assertEqual(report["seed"], 7)
        self.assertEqual(report["prompt_length"], 2)
        self.assertEqual(report["generation_strategy"], "greedy")
        self.assertGreater(report["parameters"], 0)
        self.assertEqual(report["dtype"], "float64")
        self.assertIn("python", report["environment"])
        self.assertIn("numpy", report["environment"])
        for name in (
            "forward_no_grad_tokens_per_sec",
            "infer_tokens_per_sec",
            "numpy_infer_speedup",
            "generate_cached_tokens_per_sec",
            "generate_uncached_tokens_per_sec",
            "cache_speedup",
        ):
            self.assertGreater(report[name], 0)
        for name, samples in report["samples"].items():
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(value > 0 for value in samples))
            self.assertEqual(report["summary"][name]["n"], 2)
            self.assertGreaterEqual(report["summary"][name]["sample_stdev"], 0)

    def test_numpy_speedup_pairs_matching_forward_and_infer_samples(self):
        report = run_benchmark(benchmark_args(repeats=3))
        forward = report["samples"]["forward_no_grad_seconds"]
        infer = report["samples"]["infer_seconds"]
        expected = [old / new for old, new in zip(forward, infer)]

        self.assertEqual(report["samples"]["numpy_infer_speedup"], expected)
        self.assertEqual(
            report["numpy_infer_speedup"],
            report["summary"]["numpy_infer_speedup"]["median"],
        )

    def test_validation_rejects_invalid_protocol(self):
        with self.assertRaisesRegex(ValueError, "--repeats must be positive"):
            _validate_args(benchmark_args(repeats=0))
        with self.assertRaisesRegex(ValueError, "--warmup must be non-negative"):
            _validate_args(benchmark_args(warmup=-1))

    def test_validation_rejects_non_integral_and_boolean_integer_fields(self):
        for field in (
            "vocab",
            "ctx",
            "d",
            "heads",
            "layers",
            "batch",
            "steps",
            "generate",
            "warmup",
            "repeats",
            "seed",
        ):
            with self.subTest(field=field, value=True):
                with self.assertRaises(TypeError):
                    _validate_args(benchmark_args(**{field: True}))
            with self.subTest(field=field, value=1.5):
                with self.assertRaises(TypeError):
                    _validate_args(benchmark_args(**{field: 1.5}))

    def test_numpy_integer_protocol_values_are_normalized_in_report(self):
        report = run_benchmark(
            benchmark_args(
                vocab=np.int64(32),
                ctx=np.int64(4),
                d=np.int64(8),
                heads=np.int64(2),
                layers=np.int64(1),
                batch=np.int64(1),
                steps=np.int64(1),
                generate=np.int64(1),
                warmup=np.int64(0),
                repeats=np.int64(1),
                seed=np.int64(11),
            )
        )

        for key in (
            "vocab",
            "context_len",
            "d_model",
            "heads",
            "layers",
            "batch",
            "steps",
            "generate_tokens",
            "warmup",
            "repeats",
            "seed",
        ):
            self.assertIs(type(report[key]), int)

    def test_validation_rejects_invalid_arch_and_seed_before_model_work(self):
        with self.assertRaisesRegex(TypeError, "--arch must be a string"):
            _validate_args(benchmark_args(arch=1))
        with self.assertRaisesRegex(ValueError, "unsupported --arch"):
            _validate_args(benchmark_args(arch="unknown"))
        with self.assertRaisesRegex(ValueError, "--seed must be non-negative"):
            _validate_args(benchmark_args(seed=-1))
        with self.assertRaisesRegex(ValueError, "--seed must be at most"):
            _validate_args(benchmark_args(seed=2**32))

    def test_programmatic_call_preserves_legacy_protocol_defaults(self):
        args = benchmark_args()
        del args.warmup
        del args.repeats
        report = run_benchmark(args)
        self.assertEqual(report["warmup"], 1)
        self.assertEqual(report["repeats"], 1)
        self.assertTrue(all(len(samples) == 1 for samples in report["samples"].values()))

    def test_run_benchmark_restores_caller_rng_after_success(self):
        np.random.seed(123456)
        before = np.random.get_state()

        run_benchmark(benchmark_args(repeats=1))

        self.assertTrue(_rng_equal(np.random.get_state(), before))

    def test_run_benchmark_restores_caller_rng_after_measurement_failure(self):
        np.random.seed(654321)
        before = np.random.get_state()

        with mock.patch("benchmark._time_call", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_benchmark(benchmark_args(repeats=1))

        self.assertTrue(_rng_equal(np.random.get_state(), before))

    def test_environment_metadata_avoids_identity_fields(self):
        metadata = environment_metadata()
        self.assertNotIn("hostname", metadata)
        self.assertNotIn("username", metadata)
        self.assertGreater(metadata["cpu_count"], 0)
        serialized_build = json.dumps(metadata["numpy_build"]).lower()
        self.assertNotIn("directory", serialized_build)
        self.assertNotIn("/tmp/", serialized_build)
        self.assertEqual(
            set(metadata["thread_controls"]),
            {
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            },
        )

    def test_summary_uses_sample_statistics(self):
        summary = _summarize([1.0, 2.0, 6.0])
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["sample_stdev"], 2.6457513110645907)
        with self.assertRaisesRegex(ValueError, "empty sample"):
            _summarize([])


if __name__ == "__main__":
    unittest.main()
