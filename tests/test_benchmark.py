"""Tests for the reproducible benchmark protocol and report schema."""

import argparse
import json
import os
import sys
import unittest


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
        for name, samples in report["samples"].items():
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(value > 0 for value in samples))
            self.assertEqual(report["summary"][name]["n"], 2)
            self.assertGreaterEqual(report["summary"][name]["sample_stdev"], 0)

    def test_validation_rejects_invalid_protocol(self):
        with self.assertRaisesRegex(ValueError, "--repeats must be positive"):
            _validate_args(benchmark_args(repeats=0))
        with self.assertRaisesRegex(ValueError, "--warmup must be non-negative"):
            _validate_args(benchmark_args(warmup=-1))

    def test_programmatic_call_preserves_legacy_protocol_defaults(self):
        args = benchmark_args()
        del args.warmup
        del args.repeats
        report = run_benchmark(args)
        self.assertEqual(report["warmup"], 1)
        self.assertEqual(report["repeats"], 1)
        self.assertTrue(all(len(samples) == 1 for samples in report["samples"].values()))

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
