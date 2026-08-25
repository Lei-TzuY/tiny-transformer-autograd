"""Command-line generation with explicit bounded RoPE streaming semantics."""

import argparse

import numpy as np

from engine.checkpoint import read_checkpoint, restore_checkpoint
from nn.streaming import stream_generate
from nn.transformer import GPT
from tokenizer import tokenizer_from_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate from a RoPE checkpoint with bounded streaming KV cache"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Trusted tiny-transformer checkpoint"
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Prompt text")
    prompt.add_argument("--prompt-file", help="UTF-8 prompt file")
    parser.add_argument("--tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument(
        "--strategy", choices=["sample", "greedy"], default="sample"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_streaming_checkpoint(path):
    """Restore a RoPE model and tokenizer from a training checkpoint."""
    state = read_checkpoint(path)
    metadata = state.get("metadata") or {}
    model_config = metadata.get("model_config")
    tokenizer_state = metadata.get("tokenizer")
    if model_config is None or tokenizer_state is None:
        raise ValueError(
            "streaming generation requires checkpoint metadata with model_config "
            "and tokenizer"
        )

    tokenizer = tokenizer_from_state_dict(tokenizer_state)
    model = GPT(**model_config)
    restore_checkpoint(state, model)
    model.eval()
    if model.rope is None:
        raise ValueError(
            "streaming generation requires a RoPE checkpoint "
            "(pos_encoding='rope')"
        )
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabulary sizes differ")
    return model, tokenizer


def _read_prompt(args):
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            return handle.read()
    return args.prompt


def _validate_args(args):
    if args.tokens < 0:
        raise ValueError("--tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")


def main():
    args = parse_args()
    _validate_args(args)
    model, tokenizer = load_streaming_checkpoint(args.checkpoint)
    prompt = _read_prompt(args)
    if not prompt:
        raise ValueError("generation prompt must not be empty")

    try:
        encoded = tokenizer.encode(prompt)
    except KeyError as exc:
        raise ValueError(
            f"prompt contains token not present in the tokenizer vocabulary: {exc}"
        ) from exc
    if len(encoded) == 0:
        raise ValueError("generation prompt must not be empty")

    np.random.seed(args.seed)
    visible_prompt = np.asarray(encoded[-model.context_len:], dtype=np.int64)[None, :]
    generated = stream_generate(
        model,
        visible_prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        strategy=args.strategy,
    )
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    main()
