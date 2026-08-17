"""Small benchmark utility for the NumPy GPT implementation."""

import argparse
import json
import os
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

    model.infer(idx)
    start = time.perf_counter()
    for _ in range(args.steps):
        model.infer(idx)
    infer_sec = time.perf_counter() - start

    model.generate(prompt, args.generate, strategy="greedy", use_cache=True)
    start = time.perf_counter()
    model.generate(prompt, args.generate, strategy="greedy", use_cache=True)
    cached_sec = time.perf_counter() - start

    model.generate(prompt, args.generate, strategy="greedy", use_cache=False)
    start = time.perf_counter()
    model.generate(prompt, args.generate, strategy="greedy", use_cache=False)
    uncached_sec = time.perf_counter() - start

    infer_tokens = args.steps * args.batch * args.ctx
    return {
        "arch": args.arch,
        "vocab": args.vocab,
        "context_len": args.ctx,
        "d_model": args.d,
        "heads": args.heads,
        "layers": args.layers,
        "batch": args.batch,
        "infer_tokens_per_sec": infer_tokens / infer_sec,
        "generate_cached_tokens_per_sec": args.generate / cached_sec,
        "generate_uncached_tokens_per_sec": args.generate / uncached_sec,
        "cache_speedup": uncached_sec / cached_sec if cached_sec > 0 else float("inf"),
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
    ]
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if args.arch == "llama" and (args.d // args.heads) % 2 != 0:
        raise ValueError("--arch llama needs an even head dimension (d/heads) for RoPE")


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
    print(f"  infer: {metrics['infer_tokens_per_sec']:.1f} tokens/s")
    print(f"  generate cached: {metrics['generate_cached_tokens_per_sec']:.1f} tokens/s")
    print(f"  generate uncached: {metrics['generate_uncached_tokens_per_sec']:.1f} tokens/s")
    print(f"  cache speedup: {metrics['cache_speedup']:.2f}x")


if __name__ == "__main__":
    main()
