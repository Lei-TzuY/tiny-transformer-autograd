"""Tests for the reproducible benchmark protocol and report schema."""

import argparse
import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import _validate_args, environment_metadata, run_benchmark


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
        self.assertIn("python", report["environment"])
        self.assertIn("numpy", report["environment"])
        for samples in report["samples"].values():
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(value > 0 for value in samples))

    def test_validation_rejects_invalid_protocol(self):
        with self.assertRaisesRegex(ValueError, "--repeats must be positive"):
            _validate_args(benchmark_args(repeats=0))
        with self.assertRaisesRegex(ValueError, "--warmup must be non-negative"):
            _validate_args(benchmark_args(warmup=-1))

    def test_environment_metadata_avoids_identity_fields(self):
        metadata = environment_metadata()
        self.assertNotIn("hostname", metadata)
        self.assertNotIn("username", metadata)
        self.assertGreater(metadata["cpu_count"], 0)


if __name__ == "__main__":
    unittest.main()
