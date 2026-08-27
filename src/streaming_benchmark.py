"""Reproducible strict-window versus RoPE streaming generation benchmark."""

import argparse
import json

import numpy as np

from benchmark import (
    _MAX_RANDOM_SEED,
    _strict_int,
    _summarize,
    _time_call,
    environment_metadata,
)
from nn.streaming import stream_generate
from nn.transformer import GPT


_ARCHITECTURE = {
    "norm": "rmsnorm",
    "pos_encoding": "rope",
    "ffn": "swiglu",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark strict-window and shifted RoPE-cache generation"
    )
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=32)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def run_streaming_benchmark(args):
    """Run paired strict/streaming measurements without leaking NumPy RNG state."""
    warmup, repeats, seed = _validate_args(args)
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        return _run_streaming_benchmark(args, warmup, repeats, seed)
    finally:
        np.random.set_state(rng_state)


def _run_streaming_benchmark(args, warmup, repeats, seed):
    # _validate_args explicitly accepts NumPy integer scalars. Normalize them
    # before any workload arithmetic so narrow dtypes cannot overflow (for
    # example np.int8(32) * 4 wrapping while computing d_ff).
    vocab = int(args.vocab)
    ctx = int(args.ctx)
    d_model = int(args.d)
    heads = int(args.heads)
    layers = int(args.layers)
    generate = int(args.generate)

    model = GPT(
        vocab_size=vocab,
        context_len=ctx,
        d_model=d_model,
        num_heads=heads,
        d_ff=4 * d_model,
        num_layers=layers,
        **_ARCHITECTURE,
    )
    model.eval()

    full_prompt = np.random.randint(0, vocab, size=(1, ctx))
    inside_prompt_len = max(1, ctx // 2)
    inside_prompt = full_prompt[:, :inside_prompt_len]
    inside_generate = min(generate, ctx - inside_prompt_len)

    def inside_strict_once():
        return model.generate(
            inside_prompt,
            inside_generate,
            strategy="greedy",
            use_cache=True,
        )

    def inside_streaming_once():
        return stream_generate(
            model,
            inside_prompt,
            inside_generate,
            strategy="greedy",
        )

    def saturated_strict_once():
        return model.generate(
            full_prompt,
            generate,
            strategy="greedy",
            use_cache=True,
        )

    def saturated_streaming_once():
        return stream_generate(
            model,
            full_prompt,
            generate,
            strategy="greedy",
        )

    inside_outputs_match = np.array_equal(
        inside_strict_once(),
        inside_streaming_once(),
    )
    saturated_outputs_match = np.array_equal(
        saturated_strict_once(),
        saturated_streaming_once(),
    )
    if not inside_outputs_match:
        raise RuntimeError("inside-window strict and streaming generation diverged")
    if layers == 1 and not saturated_outputs_match:
        raise RuntimeError("one-layer saturated strict and streaming generation diverged")

    for _ in range(warmup):
        inside_strict_once()
        inside_streaming_once()
        saturated_strict_once()
        saturated_streaming_once()

    inside_strict, inside_streaming = _paired_durations(
        inside_strict_once,
        inside_streaming_once,
        repeats,
    )
    saturated_strict, saturated_streaming = _paired_durations(
        saturated_strict_once,
        saturated_streaming_once,
        repeats,
    )

    samples = {}
    _add_regime_samples(
        samples,
        "inside",
        inside_strict,
        inside_streaming,
        inside_generate,
    )
    _add_regime_samples(
        samples,
        "saturated",
        saturated_strict,
        saturated_streaming,
        generate,
    )
    summaries = {name: _summarize(values) for name, values in samples.items()}

    return {
        "streaming_benchmark_schema": 1,
        "arch": "llama",
        "vocab": vocab,
        "context_len": ctx,
        "d_model": d_model,
        "heads": heads,
        "layers": layers,
        "batch": 1,
        "d_ff": 4 * d_model,
        "parameters": model.param_count(),
        "dtype": str(model.token_emb.weight.data.dtype),
        "seed": seed,
        "warmup": warmup,
        "repeats": repeats,
        "environment": environment_metadata(),
        "comparison": {
            "strict_mode": "strict_window_refill",
            "streaming_mode": "shifted_rope_cache",
            "inside_window": {
                "prompt_length": int(inside_prompt_len),
                "generate_tokens": int(inside_generate),
                "semantics_match": True,
                "outputs_match": bool(inside_outputs_match),
            },
            "saturated_window": {
                "prompt_length": ctx,
                "generate_tokens": generate,
                "semantics_match": bool(layers == 1),
                "outputs_match": bool(saturated_outputs_match),
            },
        },
        "inside_strict_tokens_per_sec": summaries[
            "inside_strict_tokens_per_sec"
        ]["median"],
        "inside_streaming_tokens_per_sec": summaries[
            "inside_streaming_tokens_per_sec"
        ]["median"],
        "inside_streaming_speedup": summaries["inside_streaming_speedup"][
            "median"
        ],
        "saturated_strict_tokens_per_sec": summaries[
            "saturated_strict_tokens_per_sec"
        ]["median"],
        "saturated_streaming_tokens_per_sec": summaries[
            "saturated_streaming_tokens_per_sec"
        ]["median"],
        "saturated_streaming_speedup": summaries[
            "saturated_streaming_speedup"
        ]["median"],
        "samples": samples,
        "summary": summaries,
    }


def _paired_durations(strict_fn, streaming_fn, repeats):
    strict = []
    streaming = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            strict.append(_time_call(strict_fn))
            streaming.append(_time_call(streaming_fn))
        else:
            streaming.append(_time_call(streaming_fn))
            strict.append(_time_call(strict_fn))
    return strict, streaming


def _add_regime_samples(samples, prefix, strict, streaming, token_count):
    samples[f"{prefix}_strict_seconds"] = strict
    samples[f"{prefix}_streaming_seconds"] = streaming
    samples[f"{prefix}_strict_tokens_per_sec"] = [
        token_count / seconds for seconds in strict
    ]
    samples[f"{prefix}_streaming_tokens_per_sec"] = [
        token_count / seconds for seconds in streaming
    ]
    samples[f"{prefix}_streaming_speedup"] = [
        strict_seconds / streaming_seconds
        for strict_seconds, streaming_seconds in zip(strict, streaming)
    ]


def _validate_args(args):
    warmup = _strict_int(getattr(args, "warmup", 1), "--warmup", minimum=0)
    repeats = _strict_int(getattr(args, "repeats", 1), "--repeats")
    vocab = _strict_int(args.vocab, "--vocab")
    ctx = _strict_int(args.ctx, "--ctx")
    d_model = _strict_int(args.d, "--d")
    heads = _strict_int(args.heads, "--heads")
    layers = _strict_int(args.layers, "--layers")
    generate = _strict_int(args.generate, "--generate")
    seed = _strict_int(args.seed, "--seed", minimum=0, maximum=_MAX_RANDOM_SEED)

    if vocab < 2:
        raise ValueError("--vocab must be at least 2")
    if ctx < 2:
        raise ValueError("--ctx must be at least 2")
    if generate < 2:
        raise ValueError("--generate must be at least 2")
    if d_model % heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if (d_model // heads) % 2 != 0:
        raise ValueError("streaming benchmark needs an even head dimension for RoPE")
    return warmup, repeats, seed


def main():
    args = parse_args()
    metrics = run_streaming_benchmark(args)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
        return

    print("Strict-window vs RoPE streaming benchmark")
    print(
        f"  shape: vocab={metrics['vocab']} ctx={metrics['context_len']} "
        f"d={metrics['d_model']} heads={metrics['heads']} "
        f"layers={metrics['layers']}"
    )
    print(
        f"  protocol: warmup={metrics['warmup']} repeats={metrics['repeats']} "
        f"seed={metrics['seed']}"
    )
    _print_regime(metrics, "inside", "inside window")
    _print_regime(metrics, "saturated", "saturated window")

    saturated = metrics["comparison"]["saturated_window"]
    if saturated["semantics_match"]:
        print("  saturated semantics: exact for this one-layer model")
    else:
        print(
            "  saturated semantics: different bounded-history policies; "
            "performance comparison only"
        )


def _print_regime(metrics, prefix, label):
    strict = metrics["summary"][f"{prefix}_strict_tokens_per_sec"]
    streaming = metrics["summary"][f"{prefix}_streaming_tokens_per_sec"]
    speedup = metrics["summary"][f"{prefix}_streaming_speedup"]
    print(
        f"  {label}: strict={strict['median']:.1f} tokens/s, "
        f"streaming={streaming['median']:.1f} tokens/s, "
        f"streaming/strict={speedup['median']:.2f}x"
    )


if __name__ == "__main__":
    main()
