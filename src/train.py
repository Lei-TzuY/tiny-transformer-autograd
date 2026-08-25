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
    encoded = []
    for document in documents:
        tokens = np.asarray(tokenizer.encode(document), dtype=np.int64)
        tokens = tokens[: context_len + 1]
        if len(tokens) >= 2:
            encoded.append(tokens)
    return encoded


def get_document_batch(documents, batch_size, pad_token=PAD_TOKEN,
                       ignore_index=IGNORE_INDEX):
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


def batch_loss(model, tokens, targets, mask=None):
    if mask is None:
        return ops.cross_entropy(model(tokens), targets)
    return ops.cross_entropy(
        model(tokens, attention_mask=mask), targets, ignore_index=IGNORE_INDEX
    )


def accumulate_document_gradients(model, sample_batch, params, grad_accum):
    if grad_accum <= 1:
        raise ValueError("document gradient accumulation requires grad_accum > 1")

    weighted_loss_sum = 0.0
    total_scored = 0
    for _ in range(grad_accum):
        batch = sample_batch()
        targets = np.asarray(batch[1])
        scored = int(np.count_nonzero(targets != IGNORE_INDEX))
        if scored == 0:
            raise ValueError("training batch contains no scored tokens")

        loss = batch_loss(model, *batch)
        loss.backward(float(scored))
        weighted_loss_sum += float(loss.data) * scored
        total_scored += scored

    for parameter in params:
        if parameter.grad is not None:
            parameter.grad /= total_scored
    return float(weighted_loss_sum / total_scored)


def evaluate_batches(
    model, sample_batch, eval_iters, weight_by_scored_tokens=False
):
    previous_mode = getattr(model, "training", True)
    model.eval()
    losses = []
    weights = []
    try:
        with no_grad():
            for _ in range(eval_iters):
                batch = sample_batch()
                losses.append(float(batch_loss(model, *batch).data))
                if weight_by_scored_tokens:
                    targets = np.asarray(batch[1])
                    scored = int(np.count_nonzero(targets != IGNORE_INDEX))
                    if scored == 0:
                        raise ValueError("evaluation batch contains no scored tokens")
                    weights.append(scored)
    finally:
        model.train(previous_mode)
    mean_loss = float(
        np.average(losses, weights=weights)
        if weight_by_scored_tokens
        else np.mean(losses)
    )
    return mean_loss, float(np.exp(min(mean_loss, 700.0)))


def evaluate(model, data, context_len, batch_size, eval_iters):
    if len(data) <= context_len:
        return None
    return evaluate_batches(
        model,
        lambda: (*get_batch(data, context_len, batch_size), None),
        eval_iters,
    )


def evaluate_documents(model, documents, batch_size, eval_iters):
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
        "--data-format", choices=["text", "lines", "jsonl"], default="text",
        help="text: one token stream, random windows; lines: one document per line; jsonl: one JSON record per line (padded, masked batches)",
    )
    parser.add_argument("--jsonl-field", type=str, default="text")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=32)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--arch", choices=["gpt", "llama"], default="gpt")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-merges", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log-jsonl", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--strategy", choices=["sample", "greedy", "beam"], default="sample")
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

    documents = None
    corpus = text
    if args.data_format != "text":
        documents = load_documents(text, args.data_format, args.jsonl_field)
        corpus = "\n".join(documents)

    checkpoint = read_checkpoint(args.resume) if args.resume else None
    metadata = checkpoint.get("metadata", {}) if checkpoint else {}
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
    else:
        encoded = encode_documents(documents, tokenizer, model_config["context_len"])
        if len(encoded) < 2:
            raise ValueError("need at least two documents of two or more tokens")
        split = min(max(1, int((1.0 - args.val_frac) * len(encoded))), len(encoded) - 1)
        train_docs, val_docs = encoded[:split], encoded[split:]

    model = GPT(**model_config)
    model.grad_checkpoint = args.grad_checkpoint
    params = model.parameters()
    optimizer_name = _resolve_optimizer_name(args.optimizer, checkpoint)
    optimizer_cls = {"adam": Adam, "adamw": AdamW}[optimizer_name]
    optimizer = optimizer_cls(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=args.iters,
        warmup_steps=min(args.warmup_iters, args.iters), min_lr=args.min_lr,
    )
    start_step = restore_checkpoint(checkpoint, model, optimizer, scheduler) if checkpoint else 0
    scheduler.total_steps = args.iters
    scheduler.warmup_steps = min(scheduler.warmup_steps, args.iters)

    if args.generate_only:
        _print_sample(model, tokenizer, corpus, args)
        return

    if train_docs is None:
        def sample_batch():
            return (*get_batch(train_data, model.context_len, args.batch), None)
        def run_eval():
            return evaluate(model, val_data, model.context_len, args.batch, args.eval_iters)
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
        optimizer.zero_grad()
        if train_docs is not None and args.grad_accum > 1:
            step_loss = accumulate_document_gradients(model, sample_batch, params, args.grad_accum)
        else:
            micro_losses = []
            for _ in range(args.grad_accum):
                loss = batch_loss(model, *sample_batch())
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
            elapsed = time.time() - started
            print(f"step {step:>5}/{args.iters}  train_loss={average:.4f}  lr={lr:.6g}  gnorm={grad_norm:.3f}  elapsed={elapsed:.1f}s")
            if args.log_jsonl:
                _append_jsonl(args.log_jsonl, {"step": step, "total_steps": args.iters, "train_loss": average, "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr, "grad_norm": grad_norm, "elapsed_sec": elapsed})

        if args.save and args.save_every and step % args.save_every == 0:
            save_checkpoint(args.save, model, optimizer, scheduler, step, _metadata(model, tokenizer))

    if args.save:
        save_checkpoint(args.save, model, optimizer, scheduler, args.iters, _metadata(model, tokenizer))

    if not args.no_sample:
        _print_sample(model, tokenizer, corpus, args)


def _print_eval(run_eval):
    validation = run_eval()
    if validation is None:
        print("[eval] skipped: validation split is too small to score")
        return
    print(f"[eval] val_loss={validation[0]:.4f}  val_ppl={validation[1]:.2f}")


def _print_sample(model, tokenizer, corpus, args):
    prompt = _prompt_array(args, tokenizer, corpus, model.context_len)
    generated = model.generate(
        prompt, max_new_tokens=args.sample, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, strategy=args.strategy,
        beam_width=args.beam_width, use_cache=not args.no_kv_cache,
    )
    print(tokenizer.decode(generated[0]))


def _prompt_array(args, tokenizer, corpus, context_len):
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
        raise ValueError(f"prompt contains token not present in the tokenizer vocabulary: {exc}") from exc
    if len(encoded) == 0:
        raise ValueError("generation prompt must not be empty")
    return np.array([encoded[-context_len:]], dtype=np.int64)


def _metadata(model, tokenizer):
    return {"model_config": model.config(), "tokenizer": tokenizer.state_dict()}


def _resolve_optimizer_name(requested, checkpoint):
    saved_type = checkpoint.get("optimizer_type") if checkpoint else None
    saved = saved_type.lower() if saved_type is not None else None
    supported = {"adam", "adamw"}
    if saved is not None and saved not in supported:
        raise ValueError(f"unsupported checkpoint optimizer type: {saved_type}")
    if requested is not None and saved is not None and requested != saved:
        raise ValueError(f"--optimizer {requested} conflicts with checkpoint optimizer {saved}")
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
    integer_names = (
        "iters", "batch", "ctx", "d", "heads", "layers", "grad_accum",
        "eval_interval", "eval_iters", "warmup_iters", "save_every",
        "bpe_merges", "lora_rank", "sample", "beam_width",
    )
    for name in integer_names:
        _validate_int_arg(args, name)
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
