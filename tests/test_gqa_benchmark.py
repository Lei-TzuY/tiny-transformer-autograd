import json

import numpy as np
import pytest

from gqa_benchmark import main, run_benchmark


def _small_report(**overrides):
    kwargs = {
        "vocab": 16,
        "context_len": 4,
        "d_model": 8,
        "heads": 4,
        "kv_heads": [4, 2, 1],
        "layers": 1,
        "batch": 1,
        "prompt_len": 2,
        "arch": "gpt",
        "warmup": 0,
        "repeats": 1,
        "seed": 7,
    }
    kwargs.update(overrides)
    return run_benchmark(**kwargs)


def test_benchmark_reports_compact_cache_bytes_and_parameter_reductions():
    report = _small_report()
    variants = {item["kv_heads"]: item for item in report["variants"]}

    assert list(variants) == [4, 2, 1]
    assert variants[4]["cache_bytes_per_token_per_batch"] == 128
    assert variants[2]["cache_bytes_per_token_per_batch"] == 64
    assert variants[1]["cache_bytes_per_token_per_batch"] == 32
    assert variants[4]["cache_ratio_vs_mha"] == 1.0
    assert variants[2]["cache_ratio_vs_mha"] == 0.5
    assert variants[1]["cache_ratio_vs_mha"] == 0.25
    assert variants[4]["cache_reduction_vs_mha"] == 0.0
    assert variants[2]["cache_reduction_vs_mha"] == 0.5
    assert variants[1]["cache_reduction_vs_mha"] == 0.75

    assert variants[4]["parameters"] > variants[2]["parameters"] > variants[1]["parameters"]
    assert variants[4]["parameter_ratio_vs_mha"] == 1.0
    assert variants[2]["parameter_ratio_vs_mha"] < 1.0
    assert variants[1]["parameter_ratio_vs_mha"] < variants[2]["parameter_ratio_vs_mha"]

    for item in variants.values():
        assert np.isfinite(item["infer_tokens_per_sec"])
        assert item["infer_tokens_per_sec"] > 0.0
        assert np.isfinite(item["cached_decode_tokens_per_sec"])
        assert item["cached_decode_tokens_per_sec"] > 0.0
        assert len(item["infer_seconds_samples"]) == 1
        assert len(item["cached_decode_seconds_samples"]) == 1


def test_benchmark_default_kv_head_set_uses_every_divisor():
    report = run_benchmark(
        vocab=16,
        context_len=4,
        d_model=12,
        heads=6,
        layers=1,
        batch=1,
        prompt_len=2,
        warmup=0,
        repeats=1,
        seed=3,
    )

    assert [item["kv_heads"] for item in report["variants"]] == [6, 3, 2, 1]


def test_benchmark_explicit_kv_heads_adds_mha_baseline_and_deduplicates():
    report = _small_report(kv_heads=[2, 2, 1])
    assert [item["kv_heads"] for item in report["variants"]] == [4, 2, 1]


def test_benchmark_is_strict_json_safe():
    report = _small_report()
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["gqa_benchmark_schema"] == 1
    assert decoded["dtype"] == "float64"


def test_benchmark_preserves_process_global_rng():
    np.random.seed(4242)
    before = np.random.get_state()
    _small_report()
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_benchmark_cli_json_output(capsys):
    report = main(
        [
            "--vocab",
            "16",
            "--ctx",
            "4",
            "--d",
            "8",
            "--heads",
            "4",
            "--kv-heads",
            "2",
            "1",
            "--layers",
            "1",
            "--batch",
            "1",
            "--prompt-len",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--seed",
            "9",
            "--json",
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert printed == report
    assert [item["kv_heads"] for item in report["variants"]] == [4, 2, 1]


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"context_len": 1}, ValueError),
        ({"d_model": 10, "heads": 4}, ValueError),
        ({"heads": 4, "kv_heads": [3]}, ValueError),
        ({"heads": 4, "kv_heads": [True]}, TypeError),
        ({"prompt_len": 4, "context_len": 4}, ValueError),
        ({"arch": "unknown"}, ValueError),
        ({"arch": "llama", "d_model": 12, "heads": 4}, ValueError),
        ({"repeats": 0}, ValueError),
        ({"warmup": -1}, ValueError),
        ({"seed": -1}, ValueError),
    ],
)
def test_benchmark_validation(kwargs, exception):
    with pytest.raises(exception):
        _small_report(**kwargs)
