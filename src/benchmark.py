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

from engine.grad_mode import no_grad
from nn.transformer import GPT
from train import _ARCH_PRESETS


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_MAX_RANDOM_SEED = 2**32 - 1


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
    """Run a deterministic benchmark without perturbing caller NumPy RNG state."""
    warmup, repeats, seed = _validate_args(args)
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        return _run_benchmark(args, warmup, repeats, seed)
    finally:
        np.random.set_state(rng_state)


def _run_benchmark(args, warmup, repeats, seed):
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

    def forward_no_grad_once():
        # This is the old validation model path: no backward graph is retained,
        # but every operator still travels through Tensor/autograd machinery.
        with no_grad():
            for _ in range(args.steps):
                model(idx)

    def infer_once():
        # Pure NumPy path used by inference and by the optimized validation path.
        for _ in range(args.steps):
            model.infer(idx)

    def generate_cached_once():
        model.generate(prompt, args.generate, strategy="greedy", use_cache=True)

    def generate_uncached_once():
        model.generate(prompt, args.generate, strategy="greedy", use_cache=False)

    for _ in range(warmup):
        forward_no_grad_once()
        infer_once()
        generate_cached_once()
        generate_uncached_once()

    forward_durations = []
    infer_durations = []
    for repeat in range(repeats):
        # Keep graph/NumPy samples adjacent and alternate their order so a
        # consistent first-run or second-run effect cannot masquerade as speedup.
        if repeat % 2 == 0:
            forward_durations.append(_time_call(forward_no_grad_once))
            infer_durations.append(_time_call(infer_once))
        else:
            infer_durations.append(_time_call(infer_once))
            forward_durations.append(_time_call(forward_no_grad_once))

    cached_durations = []
    uncached_durations = []
    for _ in range(repeats):
        # Keep the paired cache/no-cache samples adjacent so slow host drift
        # does not masquerade as a cache effect.
        cached_durations.append(_time_call(generate_cached_once))
        uncached_durations.append(_time_call(generate_uncached_once))

    infer_tokens = args.steps * args.batch * args.ctx
    forward_rates = [infer_tokens / seconds for seconds in forward_durations]
    infer_rates = [infer_tokens / seconds for seconds in infer_durations]
    numpy_infer_speedups = [
        forward / infer
        for forward, infer in zip(forward_durations, infer_durations)
    ]
    cached_rates = [args.generate / seconds for seconds in cached_durations]
    uncached_rates = [args.generate / seconds for seconds in uncached_durations]
    cache_speedups = [
        uncached / cached
        for cached, uncached in zip(cached_durations, uncached_durations)
    ]
    samples = {
        "forward_no_grad_seconds": forward_durations,
        "infer_seconds": infer_durations,
        "forward_no_grad_tokens_per_sec": forward_rates,
        "infer_tokens_per_sec": infer_rates,
        "numpy_infer_speedup": numpy_infer_speedups,
        "generate_cached_seconds": cached_durations,
        "generate_uncached_seconds": uncached_durations,
        "generate_cached_tokens_per_sec": cached_rates,
        "generate_uncached_tokens_per_sec": uncached_rates,
        "cache_speedup": cache_speedups,
    }
    summaries = {name: _summarize(values) for name, values in samples.items()}
    return {
        "benchmark_schema": 1,
        "arch": args.arch,
        "vocab": int(args.vocab),
        "context_len": int(args.ctx),
        "d_model": int(args.d),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "batch": int(args.batch),
        "d_ff": int(4 * args.d),
        "parameters": model.param_count(),
        "dtype": str(model.token_emb.weight.data.dtype),
        "prompt_length": int(prompt.shape[1]),
        "steps": int(args.steps),
        "generate_tokens": int(args.generate),
        "generation_strategy": "greedy",
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "environment": environment_metadata(),
        "forward_no_grad_tokens_per_sec": summaries[
            "forward_no_grad_tokens_per_sec"
        ]["median"],
        "infer_tokens_per_sec": summaries["infer_tokens_per_sec"]["median"],
        "numpy_infer_speedup": summaries["numpy_infer_speedup"]["median"],
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


def _strict_int(value, name, *, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _validate_args(args):
    # Before the benchmark protocol was configurable, programmatic callers
    # only supplied the model/workload fields. Preserve that API with the old
    # one-warm-up/one-measurement behavior while the CLI uses explicit defaults.
    warmup = _strict_int(getattr(args, "warmup", 1), "--warmup", minimum=0)
    repeats = _strict_int(getattr(args, "repeats", 1), "--repeats")
    for name, value in (
        ("--vocab", args.vocab),
        ("--ctx", args.ctx),
        ("--d", args.d),
        ("--heads", args.heads),
        ("--layers", args.layers),
        ("--batch", args.batch),
        ("--steps", args.steps),
        ("--generate", args.generate),
    ):
        _strict_int(value, name)
    seed = _strict_int(args.seed, "--seed", minimum=0, maximum=_MAX_RANDOM_SEED)
    if not isinstance(args.arch, str):
        raise TypeError("--arch must be a string")
    if args.arch not in _ARCH_PRESETS:
        raise ValueError(f"unsupported --arch: {args.arch!r}")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if args.arch == "llama" and (args.d // args.heads) % 2 != 0:
        raise ValueError("--arch llama needs an even head dimension (d/heads) for RoPE")
    return warmup, repeats, seed


def main():
    args = parse_args()
    metrics = run_benchmark(args)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
        return

    print("Tiny GPT benchmark")
    print(f"  arch: {metrics['arch']}")
    print(
        f"  shape: vocab={metrics['vocab']} ctx={metrics['context_len']} "
        f"d={metrics['d_model']} heads={metrics['heads']} "
        f"layers={metrics['layers']} batch={metrics['batch']}"
    )
    print(
        f"  protocol: warmup={metrics['warmup']} repeats={metrics['repeats']} "
        f"seed={metrics['seed']}"
    )
    print(
        f"  environment: Python {metrics['environment']['python']} / "
        f"NumPy {metrics['environment']['numpy']} / "
        f"{metrics['environment']['machine']}"
    )
    _print_metric(
        "forward no_grad",
        metrics,
        "forward_no_grad_tokens_per_sec",
        " tokens/s",
    )
    _print_metric("NumPy infer", metrics, "infer_tokens_per_sec", " tokens/s")
    _print_metric("NumPy infer speedup", metrics, "numpy_infer_speedup", "x")
    _print_metric(
        "generate cached", metrics, "generate_cached_tokens_per_sec", " tokens/s"
    )
    _print_metric(
        "generate uncached", metrics, "generate_uncached_tokens_per_sec", " tokens/s"
    )
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
