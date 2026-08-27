"""Train and sample from the pure-NumPy tiny GPT model."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import engine.ops as ops
from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from engine.grad_mode import no_grad
from engine.losses import label_smoothed_cross_entropy
from engine.optim import Adam, AdamW
from engine.scheduler import WarmupCosineScheduler
from nn.transformer import GPT
from tokenizer import build_tokenizer, tokenizer_from_state_dict


_BUILTIN_TEXT = """\
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them To die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to tis a consummation
Devoutly to be wished To die to sleep
To sleep perchance to dream ay there is the rub
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause there is the respect
That makes calamity of so long life
For who would bear the whips and scorns of time
The oppressors wrong the proud mans contumely
The pangs of despised love the laws delay
The insolence of office and the spurns
That patient merit of the unworthy takes
When he himself might his quietus make
With a bare bodkin who would fardels bear
To grunt and sweat under a weary life
But that the dread of something after death
The undiscovered country from whose bourn
No traveller returns puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of
Thus conscience does make cowards of us all
And thus the native hue of resolution
Is sicklied o er with the pale cast of thought
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action
"""


_ARCH_PRESETS = {
    "gpt": {"norm": "layernorm", "pos_encoding": "learned", "ffn": "gelu"},
    "llama": {"norm": "rmsnorm", "pos_encoding": "rope", "ffn": "swiglu"},
}


def clip_grad_norm_(params, max_norm=1.0):
    """Return the global L2 gradient norm and clip it without squaring overflow."""
    if isinstance(max_norm, (bool, np.bool_)) or not isinstance(
        max_norm, (int, float, np.integer, np.floating)
    ):
        raise TypeError("max_norm must be a real number")
    max_norm = float(max_norm)
    if not np.isfinite(max_norm):
        raise ValueError("max_norm must be finite")
    if max_norm < 0.0:
        raise ValueError("max_norm must be non-negative")

    # Materialise once: clipping needs two passes, and callers may supply a
    # generator rather than a reusable list.
    try:
        params = tuple(params)
    except TypeError as exc:
        raise TypeError("params must be an iterable") from exc

    gradients = []
    largest = 0.0
    for number, parameter in enumerate(params):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        if not isinstance(grad, np.ndarray):
            raise TypeError(f"gradient {number} must be a NumPy array")
        if not np.issubdtype(grad.dtype, np.number) or np.issubdtype(
            grad.dtype, np.complexfloating
        ):
            raise TypeError(f"gradient {number} must have a real numeric dtype")
        if not np.isfinite(grad).all():
            raise ValueError(f"gradient {number} must contain only finite values")
        gradients.append(grad)
        if grad.size:
            largest = max(largest, float(np.max(np.abs(grad))))

    if largest == 0.0:
        return 0.0

    # Scale before squaring. Directly evaluating grad**2 overflows for values
    # above sqrt(float_max), even though their L2 norm may still be representable.
    scaled_sumsq = 0.0
    for grad in gradients:
        scaled = grad / largest
        scaled_sumsq += float(np.sum(scaled * scaled, dtype=np.float64))
    scaled_norm = float(np.sqrt(scaled_sumsq))

    float_max = np.finfo(np.float64).max
    total = (
        float("inf")
        if scaled_norm > 0.0 and largest > float_max / scaled_norm
        else largest * scaled_norm
    )

    if max_norm > 0.0 and (not np.isfinite(total) or total > max_norm):
        # Compute the ratio in scaled coordinates as well; max_norm / total
        # would underflow to zero when the true norm exceeds float64 range.
        scale = (max_norm / largest) / scaled_norm
        for grad in gradients:
            grad *= scale
    return float(total)


def get_batch(data, context_len, batch_size):
    if len(data) <= context_len:
        raise ValueError(
            f"need more than {context_len} tokens, but only received {len(data)}"
        )
    offsets = np.random.randint(0, len(data) - context_len, size=(batch_size,))
    x = np.stack([data[i : i + context_len] for i in offsets])
    y = np.stack([data[i + 1 : i + context_len + 1] for i in offsets])
    return x, y


# ---------------------------------------------------------------------------
# Document corpora (one document per line, or per JSONL record)
# ---------------------------------------------------------------------------
# Padding is masked out of attention and its targets are ignored by the loss,
# so the id is arbitrary; 0 is simply always a valid token.
PAD_TOKEN = 0
IGNORE_INDEX = -1


def load_documents(text, data_format, jsonl_field="text"):
    """Split raw file contents into documents. Blank lines are skipped."""
    if data_format not in {"lines", "jsonl"}:
        raise ValueError("data_format must be 'lines' or 'jsonl'")

    documents = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if data_format == "lines":
            documents.append(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or jsonl_field not in record:
            raise ValueError(f"line {number} has no {jsonl_field!r} field")
        value = record[jsonl_field]
        if not isinstance(value, str):
            raise ValueError(f"line {number} field {jsonl_field!r} is not a string")
        if value.strip():
            documents.append(value)
    return documents


def encode_documents(documents, tokenizer, context_len):
    """
    Encode documents and keep the ones that can be trained on.

    Each document is truncated to ``context_len + 1`` tokens, because the input
    is the document without its last token and the target is the document
    without its first. Anything shorter than two tokens has no such pair and is
    dropped.
    """
    encoded = []
    for document in documents:
        tokens = np.asarray(tokenizer.encode(document), dtype=np.int64)
        tokens = tokens[: context_len + 1]
        if len(tokens) >= 2:
            encoded.append(tokens)
    return encoded


def get_document_batch(documents, batch_size, pad_token=PAD_TOKEN,
                       ignore_index=IGNORE_INDEX):
    """
    Sample documents into a right-padded batch.

    Returns (tokens, targets, attention_mask). Padding sits on the right, which
    is what the forward pass expects: position i is slot i. Padded keys are
    hidden by the mask and padded targets hold ``ignore_index``, so a padded
    batch trains exactly like the documents would on their own.
    """
    if not documents:
        raise ValueError("no documents are long enough to train on")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    picked = [
        documents[index]
        for index in np.random.randint(0, len(documents), size=batch_size)
    ]
    width = max(len(document) for document in picked) - 1
    tokens = np.full((batch_size, width), pad_token, dtype=np.int64)
    targets = np.full((batch_size, width), ignore_index, dtype=np.int64)
    mask = np.zeros((batch_size, width), dtype=np.int64)
    for row, document in enumerate(picked):
        length = len(document) - 1
        tokens[row, :length] = document[:-1]
        targets[row, :length] = document[1:]
        mask[row, :length] = 1
    return tokens, targets, mask


def _normalise_label_smoothing(value, *, source="label_smoothing"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{source} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{source} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{source} must be in [0, 1]")
    return value


def batch_loss(model, tokens, targets, mask=None, label_smoothing=0.0):
    """Training loss for one batch, tracked by the autograd graph.

    ``label_smoothing=0`` deliberately uses the historical combined
    ``ops.cross_entropy`` path so existing seeded training trajectories keep
    their exact arithmetic. Positive smoothing switches only the training
    objective; validation remains unsmoothed NLL/perplexity.
    """
    label_smoothing = _normalise_label_smoothing(label_smoothing)
    if mask is None:
        logits = model(tokens)
        if label_smoothing == 0.0:
            return ops.cross_entropy(logits, targets)
        return label_smoothed_cross_entropy(
            logits, targets, smoothing=label_smoothing
        )

    logits = model(tokens, attention_mask=mask)
    if label_smoothing == 0.0:
        return ops.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    return label_smoothed_cross_entropy(
        logits,
        targets,
        smoothing=label_smoothing,
        ignore_index=IGNORE_INDEX,
    )


def _cross_entropy_np(logits, targets, ignore_index=None):
    """NumPy mirror of ``ops.cross_entropy`` for inference-only validation."""
    logits = np.asarray(logits)
    if logits.ndim == 0:
        raise ValueError("cross_entropy logits must have a class dimension")
    if logits.size == 0 or logits.shape[-1] == 0:
        raise ValueError("cross_entropy inputs must be non-empty")

    targets_np = np.asarray(targets)
    expected_shape = logits.shape[:-1]
    if targets_np.shape != expected_shape:
        raise ValueError(
            "cross_entropy target shape mismatch: "
            f"expected {expected_shape}, got {targets_np.shape}"
        )
    if targets_np.size == 0:
        raise ValueError("cross_entropy targets must be non-empty")
    if not np.issubdtype(targets_np.dtype, np.integer):
        raise TypeError("cross_entropy targets must contain integers")
    if ignore_index is not None and not isinstance(ignore_index, (int, np.integer)):
        raise TypeError("cross_entropy ignore_index must be an integer or None")

    num_classes = logits.shape[-1]
    targets_flat = np.array(targets_np, dtype=np.int64, copy=True).reshape(-1)
    rows = None
    if ignore_index is not None:
        scored = targets_flat != int(ignore_index)
        if not scored.any():
            raise ValueError(
                "cross_entropy has no scored target: every position equals "
                f"ignore_index={ignore_index}"
            )
        if not scored.all():
            rows = np.flatnonzero(scored)
            targets_flat = targets_flat[rows]

    if np.any(targets_flat < 0) or np.any(targets_flat >= num_classes):
        raise ValueError(f"cross_entropy targets must be in [0, {num_classes})")

    logits_flat = logits.reshape(-1, num_classes)
    scored_logits = logits_flat if rows is None else logits_flat[rows]
    if np.isnan(scored_logits).any() or np.isposinf(scored_logits).any():
        raise ValueError(
            "cross_entropy scored logits must not contain NaN or +inf"
        )
    row_max = scored_logits.max(axis=-1, keepdims=True)
    if np.isneginf(row_max).any():
        raise ValueError(
            "cross_entropy requires at least one finite logit per scored row"
        )
    shifted = scored_logits - row_max
    normalisers = np.exp(shifted).sum(axis=-1)
    target_shifted = shifted[np.arange(targets_flat.size), targets_flat]
    return float(np.mean(np.log(normalisers) - target_shifted))


def batch_eval_loss(model, tokens, targets, mask=None):
    """Inference-only unsmoothed batch NLL for comparable validation metrics."""
    infer = getattr(model, "infer", None)
    if not callable(infer):
        with no_grad():
            return float(batch_loss(model, tokens, targets, mask, 0.0).data)

    if mask is None:
        logits, _ = infer(tokens)
        ignore_index = None
    else:
        # ``GPT.infer`` accepts generation-style masks too, while validation
        # historically follows ``GPT.forward`` and therefore requires right
        # padding. Reuse that validator before taking the fast path so the
        # public evaluation contract does not silently broaden.
        validate_mask = getattr(model, "_key_padding_bias", None)
        if callable(validate_mask):
            validate_mask(mask, np.asarray(tokens).shape)
        logits, _ = infer(tokens, attention_mask=mask)
        ignore_index = IGNORE_INDEX
    return _cross_entropy_np(logits, targets, ignore_index=ignore_index)


def accumulate_document_gradients(
    model, sample_batch, params, grad_accum, label_smoothing=0.0
):
    """Accumulate ragged document gradients as one token-weighted mean loss.

    The per-micro-batch training objective is a mean over scored targets for
    both ordinary and label-smoothed cross entropy. Averaging those means would
    give a short document batch the same influence as a long one. Instead, seed
    each scalar backward pass with its scored-token count to recover a loss sum,
    accumulate those sums, and divide leaf gradients by the total scored count.

    If a later micro-batch fails after an earlier backward pass, caller-owned
    gradient buffers are restored exactly so the failed accumulation is atomic.
    """
    if grad_accum <= 1:
        raise ValueError("document gradient accumulation requires grad_accum > 1")
    label_smoothing = _normalise_label_smoothing(label_smoothing)

    try:
        params = tuple(params)
    except TypeError as exc:
        raise TypeError("params must be an iterable") from exc

    gradient_state = []
    for parameter in params:
        original = parameter.grad
        saved = None if original is None else original.copy()
        gradient_state.append((parameter, original, saved))

    try:
        weighted_loss_sum = 0.0
        total_scored = 0
        for _ in range(grad_accum):
            batch = sample_batch()
            targets = np.asarray(batch[1])
            scored = int(np.count_nonzero(targets != IGNORE_INDEX))
            if scored == 0:
                raise ValueError("training batch contains no scored tokens")

            loss = batch_loss(
                model, *batch, label_smoothing=label_smoothing
            )
            loss.backward(float(scored))
            weighted_loss_sum += float(loss.data) * scored
            total_scored += scored

        for parameter in params:
            if parameter.grad is not None:
                parameter.grad /= total_scored
        return float(weighted_loss_sum / total_scored)
    except Exception:
        for parameter, original, saved in gradient_state:
            if original is None:
                parameter.grad = None
            else:
                original[...] = saved
                parameter.grad = original
        raise


def evaluate_batches(
    model, sample_batch, eval_iters, weight_by_scored_tokens=False
):
    """Mean unsmoothed NLL and perplexity over validation batches."""
    previous_mode = getattr(model, "training", True)
    model.eval()
    losses = []
    weights = []
    try:
        # Keep the historical validation contract even though GPT.infer is
        # pure NumPy: custom infer implementations must also run with gradients
        # disabled, and the caller's grad mode is restored by the context.
        with no_grad():
            for _ in range(eval_iters):
                batch = sample_batch()
                losses.append(batch_eval_loss(model, *batch))
                if weight_by_scored_tokens:
                    targets = np.asarray(batch[1])
                    scored = int(np.count_nonzero(targets != IGNORE_INDEX))
                    if scored == 0:
                        raise ValueError("evaluation batch contains no scored tokens")
                    weights.append(scored)
    finally:
        # A sampler/model failure must not leak eval mode into later training.
        model.train(previous_mode)
    mean_loss = float(
        np.average(losses, weights=weights)
        if weight_by_scored_tokens
        else np.mean(losses)
    )
    return mean_loss, float(np.exp(min(mean_loss, 700.0)))


def evaluate(model, data, context_len, batch_size, eval_iters):
    """Return mean validation loss and perplexity, or None for short data."""
    if len(data) <= context_len:
        return None
    return evaluate_batches(
        model,
        lambda: (*get_batch(data, context_len, batch_size), None),
        eval_iters,
    )


def evaluate_documents(model, documents, batch_size, eval_iters):
    """Token-weighted validation over documents, or None for an empty split."""
    if not documents:
        return None
    return evaluate_batches(
        model,
        lambda: get_document_batch(documents, batch_size),
        eval_iters,
        weight_by_scored_tokens=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pure-NumPy tiny GPT trainer")
    parser.add_argument("--data", type=str, default=None, help="Path to plain-text data")
    parser.add_argument(
        "--data-format",
        choices=["text", "lines", "jsonl"],
        default="text",
        help=(
            "text: one token stream, random windows; "
            "lines: one document per line; "
            "jsonl: one JSON record per line (padded, masked batches)"
        ),
    )
    parser.add_argument(
        "--jsonl-field",
        type=str,
        default="text",
        help="Field holding the document text with --data-format jsonl",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Total training iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Final cosine-decay LR")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--ctx", type=int, default=32, help="Context length")
    parser.add_argument("--d", type=int, default=64, help="Model width")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument(
        "--arch",
        choices=["gpt", "llama"],
        default="gpt",
        help="gpt: LayerNorm+learned pos+GELU; llama: RMSNorm+RoPE+SwiGLU",
    )
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-merges", type=int, default=100, help="BPE merge operations")
    parser.add_argument("--lora-rank", type=int, default=0, help="Enable LoRA with this rank")
    parser.add_argument("--lora-alpha", type=float, default=1.0, help="LoRA scaling alpha")
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default=None,
        help=(
            "adam: L2-coupled decay; adamw: decoupled decay "
            "(default: Adam for new runs, saved optimizer when resuming)"
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Micro-batches accumulated per optimizer step (simulates batch×N)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help=(
            "Training-only label smoothing in [0,1]. New runs default to 0; "
            "resume inherits a saved nonzero value unless explicitly matched. "
            "Validation loss/perplexity remain unsmoothed."
        ),
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Recompute block activations in backward (less memory, ~1 extra forward)",
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--eval-interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--eval-iters", type=int, default=10, help="Validation batches")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate and exit")
    parser.add_argument("--log-jsonl", type=str, default=None, help="Append metrics as JSONL")
    parser.add_argument("--save", type=str, default=None, help="Checkpoint output path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--save-every", type=int, default=0, help="Periodic save interval")
    parser.add_argument("--sample", type=int, default=200, help="Generated tokens at end")
    parser.add_argument("--no-sample", action="store_true", help="Skip final generation")
    parser.add_argument("--generate-only", action="store_true", help="Skip training and only sample")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation")
    parser.add_argument("--prompt-file", type=str, default=None, help="Prompt text file")
    parser.add_argument(
        "--strategy",
        choices=["sample", "greedy", "beam"],
        default="sample",
        help="Generation strategy",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    _validate_args(args)
    np.random.seed(args.seed)

    if args.data:
        with open(args.data, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = _BUILTIN_TEXT
        print("[info] Using built-in Shakespeare excerpt. Pass --data FILE for custom text.")

    # Split into documents first: the tokenizer must see the document text, not
    # the JSON scaffolding around it, or braces and quotes end up in the vocab.
    documents = None
    corpus = text
    if args.data_format != "text":
        documents = load_documents(text, args.data_format, args.jsonl_field)
        corpus = "\n".join(documents)

    checkpoint = read_checkpoint(args.resume) if args.resume else None
    metadata = checkpoint.get("metadata", {}) if checkpoint else {}
    args.label_smoothing = _resolve_label_smoothing(args.label_smoothing, metadata)
    if "tokenizer" in metadata:
        tokenizer = tokenizer_from_state_dict(metadata["tokenizer"])
    else:
        tokenizer = build_tokenizer(args.tokenizer, corpus, args.bpe_merges)

    if "model_config" in metadata:
        model_config = metadata["model_config"]
        if model_config["vocab_size"] != tokenizer.vocab_size:
            raise ValueError("checkpoint tokenizer and model vocabulary sizes differ")
    else:
        arch = _ARCH_PRESETS[args.arch]
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "context_len": args.ctx,
            "d_model": args.d,
            "num_heads": args.heads,
            "d_ff": 4 * args.d,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            **arch,
        }

    train_data = val_data = None
    train_docs = val_docs = None
    if args.generate_only:
        print(f"[info] tokenizer={tokenizer.kind}  vocab={tokenizer.vocab_size}")
    elif args.data_format == "text":
        data = tokenizer.encode(text)
        context_len = model_config["context_len"]
        split = max(context_len + 1, int((1.0 - args.val_frac) * len(data)))
        split = min(split, len(data) - context_len - 1)
        if split <= context_len:
            raise ValueError("dataset is too short for train and validation context windows")
        train_data, val_data = data[:split], data[split:]
        print(
            f"[info] tokenizer={tokenizer.kind}  vocab={tokenizer.vocab_size}  "
            f"train={len(train_data):,}  val={len(val_data):,} tokens"
        )
    else:
        encoded = encode_documents(documents, tokenizer, model_config["context_len"])
        if len(encoded) < 2:
            raise ValueError(
                "need at least two documents of two or more tokens; found "
                f"{len(encoded)} usable of {len(documents)}"
            )
        split = min(max(1, int((1.0 - args.val_frac) * len(encoded))), len(encoded) - 1)
        train_docs, val_docs = encoded[:split], encoded[split:]
        dropped = len(documents) - len(encoded)
        dropped_text = f"  dropped={dropped}" if dropped else ""
        print(
            f"[info] tokenizer={tokenizer.kind}  vocab={tokenizer.vocab_size}  "
            f"format={args.data_format}  train={len(train_docs):,}  "
            f"val={len(val_docs):,} documents{dropped_text}"
        )

    model = GPT(**model_config)
    # A runtime toggle rather than architecture, so it applies to resumed runs
    # too and is not read back from the checkpoint's model config.
    model.grad_checkpoint = args.grad_checkpoint
    params = model.parameters()
    optimizer_name = _resolve_optimizer_name(args.optimizer, checkpoint)
    optimizer_cls = {"adam": Adam, "adamw": AdamW}[optimizer_name]
    optimizer = optimizer_cls(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=args.iters,
        warmup_steps=min(args.warmup_iters, args.iters),
        min_lr=args.min_lr,
    )
    start_step = (
        restore_checkpoint(checkpoint, model, optimizer, scheduler)
        if checkpoint
        else 0
    )
    if not (args.eval_only or args.generate_only) and start_step > args.iters:
        raise ValueError(
            f"checkpoint step {start_step} exceeds requested --iters {args.iters}"
        )
    scheduler.total_steps = args.iters
    scheduler.warmup_steps = min(scheduler.warmup_steps, args.iters)
    print(f"[info] {model}")
    accum_text = f"  grad_accum={args.grad_accum}" if args.grad_accum > 1 else ""
    smoothing_text = (
        f"  label_smoothing={args.label_smoothing:g}"
        if args.label_smoothing > 0.0
        else ""
    )
    print(
        f"[info] optimizer={optimizer.__class__.__name__}  peak_lr={scheduler.base_lr:g}  "
        f"warmup={scheduler.warmup_steps}  steps={args.iters}  "
        f"resume_step={start_step}{accum_text}{smoothing_text}"
    )

    if args.generate_only:
        _print_sample(model, tokenizer, corpus, args)
        return

    # One sampler interface for both corpora: a stream yields unmasked windows,
    # a document corpus yields right-padded batches plus their mask.
    if train_docs is None:
        def sample_batch():
            return (*get_batch(train_data, model.context_len, args.batch), None)

        def run_eval():
            return evaluate(
                model, val_data, model.context_len, args.batch, args.eval_iters
            )
    else:
        def sample_batch():
            return get_document_batch(train_docs, args.batch)

        def run_eval():
            return evaluate_documents(model, val_docs, args.batch, args.eval_iters)

    if args.eval_only:
        _print_eval(run_eval)
        if not args.no_sample:
            _print_sample(model, tokenizer, corpus, args)
        return

    loss_history = []
    started = time.time()
    for step in range(start_step + 1, args.iters + 1):
        lr = scheduler.step(step - 1)

        # Gradient accumulation: backward() adds into .grad, so running
        # several micro-batches before step() simulates a larger batch without
        # retaining every micro-batch graph at once. Ragged document batches
        # need token weighting because each individual CE is already a mean.
        optimizer.zero_grad()
        if train_docs is not None and args.grad_accum > 1:
            step_loss = accumulate_document_gradients(
                model,
                sample_batch,
                params,
                args.grad_accum,
                label_smoothing=args.label_smoothing,
            )
        else:
            # Keep the historical stream path (and document grad_accum=1) in
            # exactly the same arithmetic order for seeded trajectory stability.
            micro_losses = []
            for _ in range(args.grad_accum):
                loss = batch_loss(
                    model,
                    *sample_batch(),
                    label_smoothing=args.label_smoothing,
                )
                loss.backward()
                micro_losses.append(float(loss.data))
            if args.grad_accum > 1:
                for parameter in params:
                    if parameter.grad is not None:
                        parameter.grad /= args.grad_accum
            step_loss = float(np.mean(micro_losses))

        grad_norm = clip_grad_norm_(params, max_norm=args.grad_clip)
        optimizer.step()
        loss_history.append(step_loss)

        report = step == 1 or step % args.eval_interval == 0 or step == args.iters
        if report:
            average = float(np.mean(loss_history[-args.eval_interval:]))
            validation = run_eval()
            val_loss = None if validation is None else validation[0]
            val_ppl = None if validation is None else validation[1]
            val_text = (
                "val=skipped"
                if validation is None
                else f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
            )
            elapsed = time.time() - started
            print(
                f"step {step:>5}/{args.iters}  train_loss={average:.4f}  "
                f"{val_text}  lr={lr:.6g}  gnorm={grad_norm:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )
            if args.log_jsonl:
                _append_jsonl(
                    args.log_jsonl,
                    {
                        "step": step,
                        "total_steps": args.iters,
                        "train_loss": average,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "label_smoothing": args.label_smoothing,
                        "elapsed_sec": elapsed,
                    },
                )

        if args.save and args.save_every and step % args.save_every == 0:
            save_checkpoint(
                args.save,
                model,
                optimizer,
                scheduler,
                step,
                _metadata(model, tokenizer, args.label_smoothing),
            )

    if args.save:
        save_checkpoint(
            args.save,
            model,
            optimizer,
            scheduler,
            args.iters,
            _metadata(model, tokenizer, args.label_smoothing),
        )
        print(f"[info] saved checkpoint: {args.save}")

    if not args.no_sample:
        _print_sample(model, tokenizer, corpus, args)


def _print_eval(run_eval):
    validation = run_eval()
    if validation is None:
        print("[eval] skipped: validation split is too small to score")
        return
    print(f"[eval] val_loss={validation[0]:.4f}  val_ppl={validation[1]:.2f}")


def _print_sample(model, tokenizer, corpus, args):
    print("\n" + "=" * 60)
    print("Sample generation:")
    print("=" * 60)
    prompt = _prompt_array(args, tokenizer, corpus, model.context_len)
    generated = model.generate(
        prompt,
        max_new_tokens=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        strategy=args.strategy,
        beam_width=args.beam_width,
        use_cache=not args.no_kv_cache,
    )
    print(tokenizer.decode(generated[0]))
    print("=" * 60)


def _prompt_array(args, tokenizer, corpus, context_len):
    """Build the generation prompt; the default comes from the corpus itself."""
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompt_text = handle.read()
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = corpus[: max(context_len, 10)]

    try:
        encoded = tokenizer.encode(prompt_text)
    except KeyError as exc:
        raise ValueError(
            f"prompt contains token not present in the tokenizer vocabulary: {exc}"
        ) from exc
    if len(encoded) == 0:
        raise ValueError("generation prompt must not be empty")
    return np.array([encoded[-context_len:]], dtype=np.int64)


def _metadata(model, tokenizer, label_smoothing=0.0):
    """Checkpoint metadata, retaining legacy shape when smoothing is disabled."""
    label_smoothing = _normalise_label_smoothing(
        label_smoothing, source="label_smoothing metadata"
    )
    metadata = {
        "model_config": model.config(),
        "tokenizer": tokenizer.state_dict(),
    }
    if label_smoothing != 0.0:
        metadata["training_config"] = {"label_smoothing": label_smoothing}
    return metadata


def _resolve_label_smoothing(requested, metadata):
    """Resolve the training objective without silently changing it on resume."""
    saved = None
    if metadata:
        training_config = metadata.get("training_config", {})
        if training_config is None:
            training_config = {}
        if not isinstance(training_config, dict):
            raise ValueError("checkpoint training_config metadata must be a mapping")
        if "label_smoothing" in training_config:
            saved = _normalise_label_smoothing(
                training_config["label_smoothing"],
                source="checkpoint label_smoothing",
            )

    if requested is not None:
        requested = _normalise_label_smoothing(
            requested, source="--label-smoothing"
        )
    if requested is not None and saved is not None and requested != saved:
        raise ValueError(
            f"--label-smoothing {requested:g} conflicts with checkpoint "
            f"label_smoothing {saved:g}"
        )
    if requested is not None:
        return requested
    if saved is not None:
        return saved
    return 0.0


def _resolve_optimizer_name(requested, checkpoint):
    """Choose an optimizer while preserving checkpoint update semantics."""
    saved_type = checkpoint.get("optimizer_type") if checkpoint else None
    saved = saved_type.lower() if saved_type is not None else None
    supported = {"adam", "adamw"}
    if saved is not None and saved not in supported:
        raise ValueError(f"unsupported checkpoint optimizer type: {saved_type}")
    if requested is not None and saved is not None and requested != saved:
        raise ValueError(
            f"--optimizer {requested} conflicts with checkpoint optimizer {saved}"
        )
    return saved or requested or "adam"


def _append_jsonl(path, record):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_int_arg(args, name):
    value = getattr(args, name)
    option = "--" + name.replace("_", "-")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{option} must be an integer")
    return int(value)


def _validate_real_arg(args, name):
    value = getattr(args, name)
    option = "--" + name.replace("_", "-")
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{option} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{option} must be finite")
    return value


def _validate_args(args):
    # argparse already parses these types for normal CLI use, but this function
    # is also the last fail-fast boundary before numeric values reach model,
    # optimizer, scheduler, and generation code. Explicit checks prevent NaN,
    # infinities, bool-as-int values, and malformed programmatic Namespaces from
    # slipping through comparison-only range validation.
    integer_names = (
        "iters", "batch", "ctx", "d", "heads", "layers", "grad_accum",
        "eval_interval", "eval_iters", "warmup_iters", "save_every",
        "bpe_merges", "lora_rank", "sample", "beam_width",
    )
    for name in integer_names:
        _validate_int_arg(args, name)
    # Older programmatic callers predate the CLI seed flag and may omit it.
    # Real argparse output always includes it, so validate it whenever present.
    if hasattr(args, "seed"):
        _validate_int_arg(args, "seed")
    if args.top_k is not None:
        _validate_int_arg(args, "top_k")

    real_names = (
        "val_frac", "lr", "min_lr", "dropout", "weight_decay", "grad_clip",
        "lora_alpha", "temperature",
    )
    for name in real_names:
        _validate_real_arg(args, name)
    if args.top_p is not None:
        _validate_real_arg(args, "top_p")
    if hasattr(args, "label_smoothing") and args.label_smoothing is not None:
        _normalise_label_smoothing(
            args.label_smoothing, source="--label-smoothing"
        )

    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.batch <= 0 or args.ctx <= 0:
        raise ValueError("--batch and --ctx must be positive")
    if args.d <= 0 or args.heads <= 0 or args.layers <= 0:
        raise ValueError("--d, --heads, and --layers must be positive")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
    if args.arch == "llama" and (args.d // args.heads) % 2 != 0:
        raise ValueError("--arch llama needs an even head dimension (d/heads) for RoPE")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.eval_interval <= 0 or args.eval_iters <= 0:
        raise ValueError("--eval-interval and --eval-iters must be positive")
    if args.warmup_iters < 0 or args.save_every < 0:
        raise ValueError("--warmup-iters and --save-every must be non-negative")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val-frac must be in (0, 1)")
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--lr must be positive and --min-lr must be in [0, lr]")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("--weight-decay and --grad-clip must be non-negative")
    if args.bpe_merges < 0 or args.lora_rank < 0 or args.lora_alpha <= 0:
        raise ValueError("--bpe-merges/--lora-rank must be non-negative and --lora-alpha positive")
    if args.sample < 0 or args.temperature <= 0 or args.beam_width <= 0:
        raise ValueError("--sample must be non-negative, temperature/beam-width positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.data_format not in {"text", "lines", "jsonl"}:
        raise ValueError("--data-format must be text, lines, or jsonl")
    if args.data_format == "jsonl" and not args.jsonl_field:
        raise ValueError("--jsonl-field must not be empty")
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("--prompt and --prompt-file are mutually exclusive")
    if args.eval_only and args.generate_only:
        raise ValueError("--eval-only and --generate-only are mutually exclusive")


if __name__ == "__main__":
    main()
