"""
plot_loss.py — Plot training and validation curves from a JSONL log file.

Usage:
    python plot_loss.py runs/tiny.jsonl
    python plot_loss.py runs/tiny.jsonl --out loss.png

The JSONL file is produced by train.py --log-jsonl.  Each line is a JSON
object with at least {"step": int, "train_loss": float}.  Optional fields
"val_loss" and "lr" add a validation curve and learning-rate panel when present.
"""

import argparse
import json
import math
import sys


def _record_error(path, line_number, message):
    return ValueError(f"{path}:{line_number}: {message}")


def _validate_record(record, path, line_number):
    if not isinstance(record, dict):
        raise _record_error(path, line_number, "record must be a JSON object")

    missing = [key for key in ("step", "train_loss") if key not in record]
    if missing:
        raise _record_error(
            path,
            line_number,
            f"record missing required keys: {', '.join(missing)}",
        )

    step = record["step"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise _record_error(path, line_number, "step must be an integer")

    for key in ("train_loss", "val_loss", "lr"):
        value = record.get(key)
        if value is None and key != "train_loss":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _record_error(path, line_number, f"{key} must be a real number")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise _record_error(path, line_number, f"{key} must be finite")


def _load(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _record_error(path, line_number, "invalid JSON") from exc
            _validate_record(record, path, line_number)
            records.append(record)
    if not records:
        sys.exit(f"No records found in {path}")
    return records


def _pluck(records, key):
    vals = [r.get(key) for r in records]
    steps = [r["step"] for r in records]
    pairs = [(s, v) for s, v in zip(steps, vals) if v is not None]
    if not pairs:
        return [], []
    return zip(*pairs)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from JSONL log")
    parser.add_argument("log", help="Path to .jsonl log produced by train.py --log-jsonl")
    parser.add_argument("--out", default=None, help="Save figure to this path instead of showing")
    parser.add_argument("--title", default=None, help="Figure title override")
    parser.add_argument(
        "--no-lr", action="store_true", help="Omit the learning-rate subplot"
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    try:
        records = _load(args.log)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    has_val = any(r.get("val_loss") is not None for r in records)
    has_lr = (not args.no_lr) and any(r.get("lr") is not None for r in records)
    n_panels = 1 + has_lr
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ── Loss panel ──────────────────────────────────────────────────────────
    ax = axes[0]
    train_steps, train_loss = _pluck(records, "train_loss")
    ax.plot(list(train_steps), list(train_loss), label="train loss", linewidth=1.5)
    if has_val:
        val_steps, val_loss = _pluck(records, "val_loss")
        ax.plot(
            list(val_steps),
            list(val_loss),
            label="val loss",
            linewidth=1.5,
            linestyle="--",
        )
    ax.set_ylabel("cross-entropy loss")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # ── LR panel ────────────────────────────────────────────────────────────
    if has_lr:
        lr_steps, lr_vals = _pluck(records, "lr")
        axes[1].plot(
            list(lr_steps),
            list(lr_vals),
            color="tab:orange",
            linewidth=1.5,
            label="learning rate",
        )
        axes[1].set_ylabel("learning rate")
        axes[1].set_xlabel("step")
        axes[1].legend(framealpha=0.7)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel("step")

    title = args.title or f"Training curves — {args.log}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
