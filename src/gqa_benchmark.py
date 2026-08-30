"""Benchmark GPT MHA/GQA/MQA parameter and KV-cache tradeoffs."""

import argparse
import json
import statistics
import time
from numbers import Integral

import numpy as np

from nn import GPT, convert_gpt_kv_heads


_ARCH_PRESETS = {
    "gpt": {"norm": "layernorm", "pos_encoding": "learned", "ffn": "gelu"},
    "llama": {"norm": "rmsnorm", "pos_encoding": "rope", "ffn": "swiglu"},
}


def _integer(name, value, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        relation = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return value


def _normalise_kv_heads(heads, kv_heads):
    heads = _integer("heads", heads)
    if kv_heads is None:
        values = [value for value in range(heads, 0, -1) if heads % value == 0]
    else:
        try:
            values = list(kv_heads)
        except TypeError as exc:
            raise TypeError("kv_heads must be an iterable of integers") from exc
        if not values:
            raise ValueError("kv_heads must not be empty")
        values = [_integer(f"kv_heads[{index}]", value) for index, value in enumerate(values)]
        for value in values:
            if heads % value != 0:
                raise ValueError(f"heads={heads} must be divisible by kv_heads={value}")
        # MHA is always included as the baseline for reduction ratios.
        if heads not in values:
            values.insert(0, heads)

    unique = []
    seen = set()
    for value in values:
        if heads % value != 0:
            raise ValueError(f"heads={heads} must be divisible by kv_heads={value}")
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _cache_bytes(cache):
    if isinstance(cache, np.ndarray):
        return int(cache.nbytes)
    if isinstance(cache, (list, tuple)):
        return sum(_cache_bytes(value) for value in cache)
    raise TypeError(f"unexpected KV-cache value type: {type(cache).__name__}")


def _timings(call, *, warmup, repeats):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        elapsed = time.perf_counter() - started
        samples.append(max(float(elapsed), np.finfo(np.float64).tiny))
    return samples


def run_benchmark(
    *,
    vocab=128,
    context_len=32,
    d_model=64,
    heads=8,
    kv_heads=None,
    layers=2,
    batch=1,
    prompt_len=None,
    arch="gpt",
    warmup=1,
    repeats=5,
    seed=0,
):
    """Return a strict-JSON-safe MHA/GQA/MQA comparison report."""
    vocab = _integer("vocab", vocab, minimum=2)
    context_len = _integer("context_len", context_len, minimum=2)
    d_model = _integer("d_model", d_model)
    heads = _integer("heads", heads)
    layers = _integer("layers", layers)
    batch = _integer("batch", batch)
    warmup = _integer("warmup", warmup, minimum=0)
    repeats = _integer("repeats", repeats)
    seed = _integer("seed", seed, minimum=0)
    if not isinstance(arch, str) or arch not in _ARCH_PRESETS:
        raise ValueError("arch must be 'gpt' or 'llama'")
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by heads")
    head_dim = d_model // heads
    if arch == "llama" and head_dim % 2 != 0:
        raise ValueError("llama benchmark requires an even attention head dimension")

    if prompt_len is None:
        prompt_len = min(context_len - 1, max(1, context_len // 2))
    else:
        prompt_len = _integer("prompt_len", prompt_len)
    if prompt_len >= context_len:
        raise ValueError("prompt_len must be smaller than context_len for cached decoding")

    variants = _normalise_kv_heads(heads, kv_heads)
    token_rng = np.random.RandomState(seed + 1)
    tokens = token_rng.randint(0, vocab, size=(batch, prompt_len), dtype=np.int64)
    next_token = token_rng.randint(0, vocab, size=(batch, 1), dtype=np.int64)

    rng_before = np.random.get_state()
    try:
        np.random.seed(seed)
        source = GPT(
            vocab_size=vocab,
            context_len=context_len,
            d_model=d_model,
            num_heads=heads,
            d_ff=4 * d_model,
            num_layers=layers,
            dropout=0.0,
            **_ARCH_PRESETS[arch],
        )
        source.eval()

        reports = []
        for target_kv_heads in variants:
            model = convert_gpt_kv_heads(source, target_kv_heads)
            model.eval()
            logits, cache = model.infer(tokens)
            if logits.shape != (batch, prompt_len, vocab):
                raise RuntimeError(
                    "unexpected benchmark logits shape: "
                    f"expected {(batch, prompt_len, vocab)}, got {logits.shape}"
                )

            measured_cache_bytes = _cache_bytes(cache)
            expected_cache_bytes = (
                layers
                * 2
                * batch
                * target_kv_heads
                * prompt_len
                * head_dim
                * np.dtype(np.float64).itemsize
            )
            if measured_cache_bytes != expected_cache_bytes:
                raise RuntimeError(
                    "KV-cache byte count does not match compact-cache theory: "
                    f"expected {expected_cache_bytes}, got {measured_cache_bytes}"
                )

            infer_samples = _timings(
                lambda model=model: model.infer(tokens),
                warmup=warmup,
                repeats=repeats,
            )
            decode_samples = _timings(
                lambda model=model, cache=cache: model.infer(
                    next_token, kv_cache=cache
                ),
                warmup=warmup,
                repeats=repeats,
            )
            infer_median = float(statistics.median(infer_samples))
            decode_median = float(statistics.median(decode_samples))
            parameter_count = int(sum(parameter.data.size for parameter in model.parameters()))

            reports.append(
                {
                    "kv_heads": int(target_kv_heads),
                    "query_heads": heads,
                    "query_heads_per_kv_head": heads // target_kv_heads,
                    "parameters": parameter_count,
                    "parameter_bytes": parameter_count * np.dtype(np.float64).itemsize,
                    "cache_bytes": measured_cache_bytes,
                    "cache_bytes_per_token_per_batch": (
                        measured_cache_bytes // (prompt_len * batch)
                    ),
                    "infer_tokens_per_sec": float(
                        (batch * prompt_len) / infer_median
                    ),
                    "cached_decode_tokens_per_sec": float(batch / decode_median),
                    "infer_seconds_samples": [float(value) for value in infer_samples],
                    "cached_decode_seconds_samples": [
                        float(value) for value in decode_samples
                    ],
                }
            )

        baseline = next(item for item in reports if item["kv_heads"] == heads)
        for item in reports:
            parameter_ratio = item["parameters"] / baseline["parameters"]
            cache_ratio = item["cache_bytes"] / baseline["cache_bytes"]
            item["parameter_ratio_vs_mha"] = float(parameter_ratio)
            item["parameter_reduction_vs_mha"] = float(1.0 - parameter_ratio)
            item["cache_ratio_vs_mha"] = float(cache_ratio)
            item["cache_reduction_vs_mha"] = float(1.0 - cache_ratio)

        report = {
            "gqa_benchmark_schema": 1,
            "arch": arch,
            "vocab": vocab,
            "context_len": context_len,
            "d_model": d_model,
            "heads": heads,
            "head_dim": head_dim,
            "layers": layers,
            "batch": batch,
            "prompt_len": prompt_len,
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
            "dtype": "float64",
            "variants": reports,
        }
        # Keep the public report compatible with strict JSON before returning it.
        json.dumps(report, sort_keys=True, allow_nan=False)
        return report
    finally:
        np.random.set_state(rng_before)


def _positive_cli_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_cli_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark MHA/GQA/MQA parameter and KV-cache tradeoffs"
    )
    parser.add_argument("--vocab", type=_positive_cli_int, default=128)
    parser.add_argument("--ctx", type=_positive_cli_int, default=32)
    parser.add_argument("--d", type=_positive_cli_int, default=64)
    parser.add_argument("--heads", type=_positive_cli_int, default=8)
    parser.add_argument(
        "--kv-heads",
        type=_positive_cli_int,
        nargs="+",
        default=None,
        help="KV-head counts to compare; default: every divisor of --heads",
    )
    parser.add_argument("--layers", type=_positive_cli_int, default=2)
    parser.add_argument("--batch", type=_positive_cli_int, default=1)
    parser.add_argument("--prompt-len", type=_positive_cli_int, default=None)
    parser.add_argument("--arch", choices=["gpt", "llama"], default="gpt")
    parser.add_argument("--warmup", type=_nonnegative_cli_int, default=1)
    parser.add_argument("--repeats", type=_positive_cli_int, default=5)
    parser.add_argument("--seed", type=_nonnegative_cli_int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            vocab=args.vocab,
            context_len=args.ctx,
            d_model=args.d,
            heads=args.heads,
            kv_heads=args.kv_heads,
            layers=args.layers,
            batch=args.batch,
            prompt_len=args.prompt_len,
            arch=args.arch,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        print(
            "kv_heads  params       param_red  cache_B/token  cache_red  "
            "infer_tok/s  decode_tok/s"
        )
        for item in report["variants"]:
            print(
                f"{item['kv_heads']:>8}  {item['parameters']:>10}  "
                f"{item['parameter_reduction_vs_mha'] * 100:>8.2f}%  "
                f"{item['cache_bytes_per_token_per_batch']:>13}  "
                f"{item['cache_reduction_vs_mha'] * 100:>8.2f}%  "
                f"{item['infer_tokens_per_sec']:>11.1f}  "
                f"{item['cached_decode_tokens_per_sec']:>12.1f}"
            )
    return report


if __name__ == "__main__":
    main()
