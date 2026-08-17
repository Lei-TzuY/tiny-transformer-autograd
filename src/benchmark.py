"""Small benchmark utility for the NumPy GPT implementation."""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nn.transformer import GPT
from train import _ARCH_PRESETS


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark tiny GPT inference")
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=32)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--arch", choices=["gpt", "llama"], default="gpt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def run_benchmark(args):
    warmup, repeats = _validate_args(args)
    np.random.seed(args.seed)

    model = GPT(
        vocab_size=args.vocab,
        context_len=args.ctx,
        d_model=args.d,
        num_heads=args.heads,
        d_ff=4 * args.d,
        num_layers=args.layers,
        **_ARCH_PRESETS[args.arch],
    )
    model.eval()

    idx = np.random.randint(0, args.vocab, size=(args.batch, args.ctx))
    prompt = idx[:1, : min(args.ctx, max(1, args.ctx // 2))]

    def infer_once():
        for _ in range(args.steps):
            model.infer(idx)

    def generate_cached_once():
        model.generate(prompt, args.generate, strategy="greedy", use_cache=True)

    def generate_uncached_once():
        model.generate(prompt, args.generate, strategy="greedy", use_cache=False)

    for _ in range(warmup):
        infer_once()
        generate_cached_once()
        generate_uncached_once()

    infer_durations = [_time_call(infer_once) for _ in range(repeats)]
    cached_durations = []
    uncached_durations = []
    for _ in range(repeats):
        # Keep the paired cache/no-cache samples adjacent so slow host drift
        # does not masquerade as a cache effect.
        cached_durations.append(_time_call(generate_cached_once))
        uncached_durations.append(_time_call(generate_uncached_once))

    infer_tokens = args.steps * args.batch * args.ctx
    infer_rates = [infer_tokens / seconds for seconds in infer_durations]
    cached_rates = [args.generate / seconds for seconds in cached_durations]
    uncached_rates = [args.generate / seconds for seconds in uncached_durations]
    speedups = [
        uncached / cached
        for cached, uncached in zip(cached_durations, uncached_durations)
    ]
    samples = {
        "infer_seconds": infer_durations,
        "generate_cached_seconds": cached_durations,
        "generate_uncached_seconds": uncached_durations,
        "infer_tokens_per_sec": infer_rates,
        "generate_cached_tokens_per_sec": cached_rates,
        "generate_uncached_tokens_per_sec": uncached_rates,
        "cache_speedup": speedups,
    }
    summaries = {name: _summarize(values) for name, values in samples.items()}
    return {
        "benchmark_schema": 1,
        "arch": args.arch,
        "vocab": args.vocab,
        "context_len": args.ctx,
        "d_model": args.d,
        "heads": args.heads,
        "layers": args.layers,
        "batch": args.batch,
        "d_ff": 4 * args.d,
        "parameters": model.param_count(),
        "dtype": str(model.token_emb.weight.data.dtype),
        "prompt_length": int(prompt.shape[1]),
        "steps": args.steps,
        "generate_tokens": args.generate,
        "generation_strategy": "greedy",
        "seed": args.seed,
        "warmup": warmup,
        "repeats": repeats,
        "environment": environment_metadata(),
        "infer_tokens_per_sec": summaries["infer_tokens_per_sec"]["median"],
        "generate_cached_tokens_per_sec": summaries[
            "generate_cached_tokens_per_sec"
        ]["median"],
        "generate_uncached_tokens_per_sec": summaries[
            "generate_uncached_tokens_per_sec"
        ]["median"],
        "cache_speedup": summaries["cache_speedup"]["median"],
        "samples": samples,
        "summary": summaries,
    }


def _time_call(fn):
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        raise RuntimeError("benchmark timer did not advance")
    return elapsed


def environment_metadata():
    """Return reproducibility metadata without exposing hostnames or user paths."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy_build": _numpy_build_metadata(),
        "thread_controls": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
    }


def _numpy_build_metadata():
    """Return a path-free description of NumPy's BLAS and SIMD build."""
    result = {"blas": {}, "simd": {}}
    config = getattr(np.__config__, "CONFIG", None)
    if isinstance(config, dict):
        dependencies = config.get("Build Dependencies", {})
        blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
        if isinstance(blas, dict):
            for source, target in (
                ("name", "name"),
                ("version", "version"),
                ("openblas configuration", "configuration"),
            ):
                value = blas.get(source)
                if isinstance(value, (str, int, float, bool)):
                    result["blas"][target] = value
        simd = config.get("SIMD Extensions", {})
        if isinstance(simd, dict):
            for name in ("baseline", "found", "not found"):
                value = simd.get(name)
                if isinstance(value, (list, tuple)):
                    result["simd"][name.replace(" ", "_")] = list(value)
        return result

    # NumPy 1.x exposes a less structured API. Keep only library names and
    # macros; directory fields can reveal local build paths and are omitted.
    get_info = getattr(np.__config__, "get_info", None)
    if get_info is not None:
        legacy = get_info("blas_opt_info") or get_info("blas_info")
        if isinstance(legacy, dict):
            libraries = legacy.get("libraries")
            if isinstance(libraries, (list, tuple)):
                result["blas"]["libraries"] = list(libraries)
            macros = legacy.get("define_macros")
            if isinstance(macros, (list, tuple)):
                result["blas"]["define_macros"] = [list(item) for item in macros]
    return result


def _summarize(samples):
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "sample_stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def _validate_args(args):
    # Before the benchmark protocol was configurable, programmatic callers
    # only supplied the model/workload fields. Preserve that API with the old
    # one-warm-up/one-measurement behavior while the CLI uses its explicit
    # defaults from parse_args().
    warmup = getattr(args, "warmup", 1)
    repeats = getattr(args, "repeats", 1)
    positive = [
        ("--vocab", args.vocab),
        ("--ctx", args.ctx),
        ("--d", args.d),
        ("--heads", args.heads),
        ("--layers", args.layers),
        ("--batch", args.batch),
        ("--steps", args.steps),
        ("--generate", args.generate),
        ("--repeats", repeats),
    ]
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if args.arch == "llama" and (args.d // args.heads) % 2 != 0:
        raise ValueError("--arch llama needs an even head dimension (d/heads) for RoPE")
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")
    return warmup, repeats


def main():
    args = parse_args()
    metrics = run_benchmark(args)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
        return

    print("Tiny GPT benchmark")
    print(f"  arch: {metrics['arch']}")
    print(f"  shape: vocab={metrics['vocab']} ctx={metrics['context_len']} "
          f"d={metrics['d_model']} heads={metrics['heads']} layers={metrics['layers']} "
          f"batch={metrics['batch']}")
    print(f"  protocol: warmup={metrics['warmup']} repeats={metrics['repeats']} seed={metrics['seed']}")
    print(f"  environment: Python {metrics['environment']['python']} / "
          f"NumPy {metrics['environment']['numpy']} / {metrics['environment']['machine']}")
    _print_metric("infer", metrics, "infer_tokens_per_sec", " tokens/s")
    _print_metric("generate cached", metrics, "generate_cached_tokens_per_sec", " tokens/s")
    _print_metric("generate uncached", metrics, "generate_uncached_tokens_per_sec", " tokens/s")
    _print_metric("cache speedup", metrics, "cache_speedup", "x")


def _print_metric(label, metrics, key, suffix):
    summary = metrics["summary"][key]
    print(
        f"  {label}: {summary['median']:.1f}{suffix} median "
        f"(min={summary['min']:.1f}, max={summary['max']:.1f}, "
        f"sample_stdev={summary['sample_stdev']:.1f}, n={summary['n']})"
    )


if __name__ == "__main__":
    main()
