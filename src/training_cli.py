"""GQA-aware console adapters for the existing tiny GPT trainer.

The historical :mod:`train` module owns the training loop.  This module adds one
architecture option at the packaged console boundary without duplicating that
loop: ``--kv-heads`` is stripped before delegating to ``train.main()`` and is
injected only when the GPT is constructed.

On resume, checkpoint ``metadata['model_config']`` remains authoritative.  An
explicit ``--kv-heads`` must match the checkpoint's effective KV-head count;
omitting the option preserves the checkpoint architecture exactly.
"""

import argparse
from collections.abc import Mapping
import os
import sys

import numpy as np

import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


_UNSET = object()


def _positive_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _extract_kv_heads(argv):
    """Return ``(argv_without_option, requested_kv_heads)``.

    A tiny ``parse_known_args`` parser gives the new option normal argparse
    syntax (including ``--kv-heads=2``) while leaving every historical trainer
    argument and its ordering untouched for ``train.parse_args``.
    """
    if not argv:
        argv = ["tiny-train"]
    parser = argparse.ArgumentParser(
        prog=os.path.basename(os.fspath(argv[0])),
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--kv-heads", type=_positive_int, default=None)
    known, remaining = parser.parse_known_args(list(argv[1:]))
    return [argv[0], *remaining], known.kv_heads


def _checkpoint_kv_heads(state):
    """Return a checkpoint's effective KV-head count, or ``None`` if unknown."""
    if not isinstance(state, Mapping):
        return None
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    config = metadata.get("model_config")
    if not isinstance(config, Mapping):
        return None

    query_heads = config.get("num_heads", _UNSET)
    if query_heads is _UNSET:
        return None
    if isinstance(query_heads, (bool, np.bool_)) or not isinstance(
        query_heads, (int, np.integer)
    ):
        raise TypeError("checkpoint model_config num_heads must be an integer")
    query_heads = int(query_heads)
    if query_heads <= 0:
        raise ValueError("checkpoint model_config num_heads must be positive")

    kv_heads = config.get("num_kv_heads", query_heads)
    if isinstance(kv_heads, (bool, np.bool_)) or not isinstance(
        kv_heads, (int, np.integer)
    ):
        raise TypeError("checkpoint model_config num_kv_heads must be an integer")
    kv_heads = int(kv_heads)
    if kv_heads <= 0:
        raise ValueError("checkpoint model_config num_kv_heads must be positive")
    if query_heads % kv_heads != 0:
        raise ValueError(
            "checkpoint model_config num_heads must be divisible by num_kv_heads"
        )
    return kv_heads


def _gpt_factory(original_gpt, requested_kv_heads):
    if requested_kv_heads is None:
        return original_gpt

    def construct(*args, **kwargs):
        # train.main constructs GPT from a model-config mapping with keyword
        # arguments.  Keep a defensive error if that internal contract changes,
        # rather than silently applying the option to the wrong positional slot.
        if args:
            raise RuntimeError(
                "GQA training adapter expected keyword-only GPT construction"
            )
        if "num_heads" not in kwargs:
            raise RuntimeError("GPT model config is missing num_heads")
        query_heads = kwargs["num_heads"]
        if isinstance(query_heads, (bool, np.bool_)) or not isinstance(
            query_heads, (int, np.integer)
        ):
            raise TypeError("GPT model config num_heads must be an integer")
        query_heads = int(query_heads)
        if query_heads <= 0:
            raise ValueError("GPT model config num_heads must be positive")
        if query_heads % requested_kv_heads != 0:
            raise ValueError(
                f"--kv-heads {requested_kv_heads} must divide model num_heads "
                f"{query_heads}"
            )

        existing = kwargs.get("num_kv_heads")
        if existing is not None and int(existing) != requested_kv_heads:
            raise ValueError(
                f"--kv-heads {requested_kv_heads} conflicts with model config "
                f"num_kv_heads {existing}"
            )
        configured = dict(kwargs)
        configured["num_kv_heads"] = requested_kv_heads
        return original_gpt(**configured)

    return construct


def _run(*, checkpoint_reader, checkpoint_writer):
    original_argv = sys.argv
    stripped_argv, requested_kv_heads = _extract_kv_heads(original_argv)
    help_requested = any(value in {"-h", "--help"} for value in stripped_argv[1:])

    original_gpt = train.GPT
    original_reader = train.read_checkpoint
    original_writer = train.save_checkpoint

    def read_checkpoint(path):
        state = checkpoint_reader(path)
        effective = _checkpoint_kv_heads(state)
        if (
            requested_kv_heads is not None
            and effective is not None
            and requested_kv_heads != effective
        ):
            raise ValueError(
                f"--kv-heads {requested_kv_heads} conflicts with checkpoint "
                f"num_kv_heads {effective}; migrate the checkpoint explicitly "
                "before resuming"
            )
        return state

    sys.argv = stripped_argv
    train.GPT = _gpt_factory(original_gpt, requested_kv_heads)
    train.read_checkpoint = read_checkpoint
    train.save_checkpoint = checkpoint_writer
    try:
        try:
            return train.main()
        except SystemExit as exc:
            if help_requested and exc.code in (None, 0):
                print("\nadditional attention architecture option:")
                print(
                    "  --kv-heads KV_HEADS   key/value heads for a new model; "
                    "1 selects MQA. On resume, an explicit value must match "
                    "the checkpoint architecture."
                )
            raise
    finally:
        sys.argv = original_argv
        train.GPT = original_gpt
        train.read_checkpoint = original_reader
        train.save_checkpoint = original_writer


def main():
    """Run the trusted-pickle trainer with optional ``--kv-heads``."""
    return _run(
        checkpoint_reader=train.read_checkpoint,
        checkpoint_writer=train.save_checkpoint,
    )


def safe_main():
    """Run the non-executable safe-checkpoint trainer with ``--kv-heads``."""
    return _run(
        checkpoint_reader=read_safe_checkpoint,
        checkpoint_writer=save_safe_checkpoint,
    )


if __name__ == "__main__":
    main()
