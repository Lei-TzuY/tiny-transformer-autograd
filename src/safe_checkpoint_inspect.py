"""Inspect non-executable safe checkpoints without restoring training state."""

import argparse
import json
from collections.abc import Mapping

import numpy as np

from engine.safe_checkpoint import read_safe_checkpoint


def _model_summary(model_state):
    """Return deterministic tensor metadata for one saved model state."""
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint model state must be a mapping")

    tensors = []
    scalar_count = 0
    for name in sorted(model_state):
        if not isinstance(name, str):
            raise TypeError("checkpoint model state keys must be strings")
        value = model_state[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"checkpoint model state value for {name} must be a NumPy array")
        scalar_count += int(value.size)
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "scalars": int(value.size),
            }
        )

    return {
        "tensor_count": len(tensors),
        "scalar_count": scalar_count,
        "tensors": tensors,
    }


def summarize_safe_checkpoint(state):
    """Build a JSON-safe summary from a decoded safe-checkpoint state."""
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a mapping")

    metadata = state.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    metadata_keys = []
    for key in metadata:
        if not isinstance(key, str):
            raise TypeError("checkpoint metadata keys must be strings")
        metadata_keys.append(key)

    return {
        "format_version": int(state.get("format_version", 1)),
        "step": int(state.get("step", 0)),
        "optimizer_type": state.get("optimizer_type"),
        "has_optimizer": state.get("optimizer") is not None,
        "has_scheduler": state.get("scheduler") is not None,
        "metadata_keys": sorted(metadata_keys),
        "model": _model_summary(state.get("model")),
    }


def _print_human(summary, *, show_tensors):
    model = summary["model"]
    optimizer = summary["optimizer_type"] if summary["has_optimizer"] else "none"
    print(f"format_version: {summary['format_version']}")
    print(f"step: {summary['step']}")
    print(f"optimizer: {optimizer}")
    print(f"scheduler: {'present' if summary['has_scheduler'] else 'none'}")
    print(f"model_tensors: {model['tensor_count']}")
    print(f"model_scalars: {model['scalar_count']}")
    if summary["metadata_keys"]:
        print("metadata_keys: " + ", ".join(summary["metadata_keys"]))
    else:
        print("metadata_keys: none")

    if show_tensors:
        for tensor in model["tensors"]:
            shape = "x".join(str(size) for size in tensor["shape"]) or "scalar"
            print(
                f"tensor {tensor['name']}: shape={shape}, "
                f"dtype={tensor['dtype']}, scalars={tensor['scalars']}"
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect a non-executable tiny-transformer safe checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to a .safe.npz checkpoint")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="List each saved model tensor in human-readable output",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_safe_checkpoint(read_safe_checkpoint(args.checkpoint))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    else:
        _print_human(summary, show_tensors=args.tensors)
    return 0


if __name__ == "__main__":
    main()
