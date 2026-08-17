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
    _validate_args(args)
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

    for _ in range(args.warmup):
        infer_once()
        generate_cached_once()
        generate_uncached_once()

    infer_durations = [_time_call(infer_once) for _ in range(args.repeats)]
    cached_durations = []
    uncached_durations = []
    for _ in range(args.repeats):
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
    return {
        "benchmark_schema": 1,
        "arch": args.arch,
        "vocab": args.vocab,
        "context_len": args.ctx,
        "d_model": args.d,
        "heads": args.heads,
        "layers": args.layers,
        "batch": args.batch,
        "steps": args.steps,
        "generate_tokens": args.generate,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "environment": environment_metadata(),
        "infer_tokens_per_sec": statistics.median(infer_rates),
        "generate_cached_tokens_per_sec": statistics.median(cached_rates),
        "generate_uncached_tokens_per_sec": statistics.median(uncached_rates),
        "cache_speedup": statistics.median(speedups),
        "samples": {
            "infer_tokens_per_sec": infer_rates,
            "generate_cached_tokens_per_sec": cached_rates,
            "generate_uncached_tokens_per_sec": uncached_rates,
            "cache_speedup": speedups,
        },
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
    }


def _validate_args(args):
    positive = [
        ("--vocab", args.vocab),
        ("--ctx", args.ctx),
        ("--d", args.d),
        ("--heads", args.heads),
        ("--layers", args.layers),
        ("--batch", args.batch),
        ("--steps", args.steps),
        ("--generate", args.generate),
        ("--repeats", args.repeats),
    ]
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if args.arch == "llama" and (args.d // args.heads) % 2 != 0:
        raise ValueError("--arch llama needs an even head dimension (d/heads) for RoPE")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")


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
    samples = metrics["samples"][key]
    print(
        f"  {label}: {metrics[key]:.1f}{suffix} median "
        f"(min={min(samples):.1f}, max={max(samples):.1f}, n={len(samples)})"
    )


if __name__ == "__main__":
    main()
