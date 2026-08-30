"""Command-line checkpoint migration between MHA, GQA, and MQA."""

import argparse
import json
import os

from nn import convert_gpt_checkpoint_file


_FORMAT_SUFFIXES = {
    ".npz": "safe",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".ckpt": "pickle",
}


def _positive_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _infer_format(path):
    try:
        suffix = os.path.splitext(os.fspath(path))[1].lower()
    except TypeError as exc:
        raise TypeError("checkpoint path must be path-like") from exc
    if suffix not in _FORMAT_SUFFIXES:
        raise ValueError(
            "cannot infer checkpoint format from extension; use "
            "--source-format/--destination-format"
        )
    return _FORMAT_SUFFIXES[suffix]


def _resolve_format(requested, path, *, fallback=None):
    if requested is None or requested == "auto":
        if fallback is not None:
            return fallback
        return _infer_format(path)
    return requested


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a tiny-transformer GPT checkpoint between MHA, GQA, and MQA"
    )
    parser.add_argument("source", help="Input checkpoint path")
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help="Output checkpoint path (default: convert source in place)",
    )
    parser.add_argument(
        "--kv-heads",
        type=_positive_int,
        required=True,
        help="Target number of key/value heads; 1 selects MQA",
    )
    parser.add_argument(
        "--source-format",
        choices=["auto", "pickle", "safe"],
        default="auto",
        help="Input encoding (default: infer from extension)",
    )
    parser.add_argument(
        "--destination-format",
        choices=["auto", "pickle", "safe"],
        default="auto",
        help=(
            "Output encoding (default: same format for in-place conversion, "
            "otherwise infer from destination extension)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON report",
    )
    return parser


def convert_from_args(args):
    destination = args.source if args.destination is None else args.destination
    source_format = _resolve_format(args.source_format, args.source)
    destination_format = _resolve_format(
        args.destination_format,
        destination,
        fallback=source_format if args.destination is None else None,
    )

    converted = convert_gpt_checkpoint_file(
        args.source,
        destination,
        args.kv_heads,
        source_format=source_format,
        destination_format=destination_format,
    )
    metadata = converted.get("metadata", {})
    model_config = metadata.get("model_config", {})
    query_heads = int(model_config["num_heads"])
    target_kv_heads = int(model_config.get("num_kv_heads", query_heads))
    history = metadata.get("_tiny_transformer_migrations", [])
    latest = history[-1] if history else None
    migration_applied = bool(
        latest
        and latest.get("kind") == "gpt_kv_heads"
        and latest.get("target_num_kv_heads") == target_kv_heads
    )

    return {
        "source_format": source_format,
        "destination_format": destination_format,
        "in_place": os.path.abspath(os.fspath(args.source))
        == os.path.abspath(os.fspath(destination)),
        "checkpoint_step": int(converted.get("step", 0)),
        "optimizer_type": converted.get("optimizer_type"),
        "query_heads": query_heads,
        "kv_heads": target_kv_heads,
        "kv_cache_head_ratio": target_kv_heads / query_heads,
        "migration_applied": migration_applied,
        "optimizer_state_reset": (
            latest.get("optimizer_state") == "reset" if migration_applied else False
        ),
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = convert_from_args(args)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        mode = "in-place" if report["in_place"] else "copy"
        reset = "yes" if report["optimizer_state_reset"] else "no"
        print(
            "converted GPT checkpoint: "
            f"query_heads={report['query_heads']}  kv_heads={report['kv_heads']}  "
            f"cache_head_ratio={report['kv_cache_head_ratio']:.4g}  "
            f"optimizer_reset={reset}  mode={mode}"
        )
    return report


if __name__ == "__main__":
    main()
