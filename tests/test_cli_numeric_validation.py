"""Fail-fast contracts for numeric tiny-train arguments."""

import os
import sys
from argparse import Namespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import _validate_args, parse_args


def _valid_args(**overrides):
    values = {
        "iters": 1,
        "batch": 1,
        "ctx": 4,
        "d": 8,
        "heads": 2,
        "layers": 1,
        "arch": "gpt",
        "data_format": "text",
        "jsonl_field": "text",
        "optimizer": "adam",
        "grad_accum": 1,
        "eval_interval": 1,
        "eval_iters": 1,
        "warmup_iters": 0,
        "save_every": 0,
        "val_frac": 0.1,
        "lr": 1e-3,
        "min_lr": 0.0,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "bpe_merges": 0,
        "lora_rank": 0,
        "lora_alpha": 1.0,
        "sample": 0,
        "temperature": 1.0,
        "beam_width": 1,
        "top_k": None,
        "top_p": None,
        "seed": 42,
        "prompt": None,
        "prompt_file": None,
        "eval_only": False,
        "generate_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "name",
    [
        "val_frac",
        "lr",
        "min_lr",
        "dropout",
        "weight_decay",
        "grad_clip",
        "lora_alpha",
        "temperature",
        "top_p",
    ],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_real_arguments(name, bad):
    args = _valid_args(**{name: bad})
    with pytest.raises(ValueError, match=rf"--{name.replace('_', '-')} must be finite"):
        _validate_args(args)


@pytest.mark.parametrize(
    "name",
    [
        "iters",
        "batch",
        "ctx",
        "d",
        "heads",
        "layers",
        "grad_accum",
        "eval_interval",
        "eval_iters",
        "warmup_iters",
        "save_every",
        "bpe_merges",
        "lora_rank",
        "sample",
        "beam_width",
        "seed",
        "top_k",
    ],
)
@pytest.mark.parametrize("bad", [True, 1.5])
def test_rejects_bool_and_fractional_integer_arguments(name, bad):
    args = _valid_args(**{name: bad})
    with pytest.raises(TypeError, match=rf"--{name.replace('_', '-')} must be an integer"):
        _validate_args(args)


@pytest.mark.parametrize(
    "name",
    [
        "val_frac",
        "lr",
        "min_lr",
        "dropout",
        "weight_decay",
        "grad_clip",
        "lora_alpha",
        "temperature",
        "top_p",
    ],
)
def test_rejects_bool_for_real_arguments(name):
    args = _valid_args(**{name: True})
    with pytest.raises(TypeError, match=rf"--{name.replace('_', '-')} must be a real number"):
        _validate_args(args)


def test_accepts_numpy_numeric_scalars():
    args = _valid_args(
        iters=np.int64(2),
        batch=np.int32(1),
        seed=np.int64(7),
        top_k=np.int32(3),
        lr=np.float32(1e-3),
        min_lr=np.float64(0.0),
        val_frac=np.float32(0.2),
        grad_clip=np.float64(1.0),
        top_p=np.float32(0.9),
    )
    _validate_args(args)


def test_actual_cli_nan_is_rejected_after_argparse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiny-train", "--lr", "nan"])
    args = parse_args()
    assert np.isnan(args.lr)
    with pytest.raises(ValueError, match="--lr must be finite"):
        _validate_args(args)


def test_range_contracts_remain_unchanged_after_type_preflight():
    with pytest.raises(ValueError, match="--val-frac must be in"):
        _validate_args(_valid_args(val_frac=1.0))
    with pytest.raises(ValueError, match="--top-k must be positive"):
        _validate_args(_valid_args(top_k=0))
    with pytest.raises(ValueError, match="--d must be divisible"):
        _validate_args(_valid_args(d=7, heads=2))
