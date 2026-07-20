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
from engine.optim import Adam
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


def clip_grad_norm_(params, max_norm=1.0):
    total = np.sqrt(sum(np.sum(p.grad ** 2) for p in params if p.grad is not None))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad *= scale
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


def evaluate(model, data, context_len, batch_size, eval_iters):
    """Return mean validation loss and perplexity, or None for short data."""
    if len(data) <= context_len:
        return None
    previous_mode = getattr(model, "training", True)
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, context_len, batch_size)
        logits = model(x)
        batch, sequence_len, vocab = logits.shape
        loss = ops.cross_entropy(
            ops.reshape(logits, (batch * sequence_len, vocab)), y.reshape(-1)
        )
        losses.append(float(loss.data))
    model.train(previous_mode)
    mean_loss = float(np.mean(losses))
    return mean_loss, float(np.exp(min(mean_loss, 700.0)))


def parse_args():
    parser = argparse.ArgumentParser(description="Pure-NumPy tiny GPT trainer")
    parser.add_argument("--data", type=str, default=None, help="Path to plain-text data")
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
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-merges", type=int, default=100, help="BPE merge operations")
    parser.add_argument("--lora-rank", type=int, default=0, help="Enable LoRA with this rank")
    parser.add_argument("--lora-alpha", type=float, default=1.0, help="LoRA scaling alpha")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Adam weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
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

    checkpoint = read_checkpoint(args.resume) if args.resume else None
    metadata = checkpoint.get("metadata", {}) if checkpoint else {}
    if "tokenizer" in metadata:
        tokenizer = tokenizer_from_state_dict(metadata["tokenizer"])
    else:
        tokenizer = build_tokenizer(args.tokenizer, text, args.bpe_merges)

    if "model_config" in metadata:
        model_config = metadata["model_config"]
        if model_config["vocab_size"] != tokenizer.vocab_size:
            raise ValueError("checkpoint tokenizer and model vocabulary sizes differ")
    else:
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
        }

    if args.generate_only:
        train_data = val_data = None
        print(f"[info] tokenizer={tokenizer.kind}  vocab={tokenizer.vocab_size}")
    else:
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

    model = GPT(**model_config)
    params = model.parameters()
    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
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
    scheduler.total_steps = args.iters
    scheduler.warmup_steps = min(scheduler.warmup_steps, args.iters)
    print(f"[info] {model}")
    print(
        f"[info] optimizer=Adam  peak_lr={scheduler.base_lr:g}  "
        f"warmup={scheduler.warmup_steps}  steps={args.iters}  resume_step={start_step}"
    )

    if args.generate_only:
        _print_sample(model, tokenizer, text, args)
        return

    if args.eval_only:
        _print_eval(model, val_data, args)
        if not args.no_sample:
            _print_sample(model, tokenizer, text, args)
        return

    loss_history = []
    started = time.time()
    for step in range(start_step + 1, args.iters + 1):
        lr = scheduler.step(step - 1)
        x, y = get_batch(train_data, model.context_len, args.batch)
        logits = model(x)
        batch, sequence_len, vocab = logits.shape
        loss = ops.cross_entropy(
            ops.reshape(logits, (batch * sequence_len, vocab)), y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(params, max_norm=args.grad_clip)
        optimizer.step()
        loss_history.append(float(loss.data))

        report = step == 1 or step % args.eval_interval == 0 or step == args.iters
        if report:
            average = float(np.mean(loss_history[-args.eval_interval:]))
            validation = evaluate(
                model, val_data, model.context_len, args.batch, args.eval_iters
            )
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
                        "elapsed_sec": elapsed,
                    },
                )

        if args.save and args.save_every and step % args.save_every == 0:
            save_checkpoint(args.save, model, optimizer, scheduler, step, _metadata(model, tokenizer))

    if args.save:
        save_checkpoint(
            args.save, model, optimizer, scheduler, args.iters, _metadata(model, tokenizer)
        )
        print(f"[info] saved checkpoint: {args.save}")

    if not args.no_sample:
        _print_sample(model, tokenizer, text, args)


def _print_eval(model, val_data, args):
    validation = evaluate(model, val_data, model.context_len, args.batch, args.eval_iters)
    if validation is None:
        print("[eval] skipped: validation split is shorter than context length")
        return
    print(f"[eval] val_loss={validation[0]:.4f}  val_ppl={validation[1]:.2f}")


def _print_sample(model, tokenizer, text, args):
    print("\n" + "=" * 60)
    print("Sample generation:")
    print("=" * 60)
    prompt = _prompt_array(args, tokenizer, text, model.context_len)
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


def _prompt_array(args, tokenizer, text, context_len):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompt_text = handle.read()
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = text[: max(context_len, 10)]

    try:
        encoded = tokenizer.encode(prompt_text)
    except KeyError as exc:
        raise ValueError(
            f"prompt contains token not present in the tokenizer vocabulary: {exc}"
        ) from exc
    if len(encoded) == 0:
        raise ValueError("generation prompt must not be empty")
    return np.array([encoded[-context_len:]], dtype=np.int64)


def _metadata(model, tokenizer):
    return {
        "model_config": model.config(),
        "tokenizer": tokenizer.state_dict(),
    }


def _append_jsonl(path, record):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_args(args):
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.batch <= 0 or args.ctx <= 0:
        raise ValueError("--batch and --ctx must be positive")
    if args.d <= 0 or args.heads <= 0 or args.layers <= 0:
        raise ValueError("--d, --heads, and --layers must be positive")
    if args.d % args.heads != 0:
        raise ValueError("--d must be divisible by --heads")
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
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("--prompt and --prompt-file are mutually exclusive")
    if args.eval_only and args.generate_only:
        raise ValueError("--eval-only and --generate-only are mutually exclusive")


if __name__ == "__main__":
    main()
