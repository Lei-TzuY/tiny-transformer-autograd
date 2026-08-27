"""Validation regressions for the source-checkout loss plotting utility."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "plot_loss.py"
SPEC = importlib.util.spec_from_file_location("plot_loss", MODULE_PATH)
plot_loss = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_loss)


def _write(tmp_path, text):
    path = tmp_path / "train.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_accepts_valid_training_records(tmp_path):
    path = _write(
        tmp_path,
        '{"step": 1, "train_loss": 2.5, "lr": 0.001}\n'
        '\n'
        '{"step": 2, "train_loss": 2.0, "val_loss": 2.2}\n',
    )

    records = plot_loss._load(path)

    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["lr"] == 0.001
    assert records[1]["val_loss"] == 2.2


@pytest.mark.parametrize(
    "text,match",
    [
        ('{"step": 1,\n', r"train\.jsonl:1: invalid JSON"),
        ('[]\n', r"train\.jsonl:1: record must be a JSON object"),
        ('{"train_loss": 1.0}\n', r"train\.jsonl:1: record missing required keys: step"),
        ('{"step": 1}\n', r"train\.jsonl:1: record missing required keys: train_loss"),
        ('{"step": true, "train_loss": 1.0}\n', r"train\.jsonl:1: step must be an integer"),
        ('{"step": 1, "train_loss": "bad"}\n', r"train\.jsonl:1: train_loss must be a real number"),
        ('{"step": 1, "train_loss": NaN}\n', r"train\.jsonl:1: train_loss must be finite"),
        ('{"step": 1, "train_loss": 1.0, "val_loss": Infinity}\n', r"train\.jsonl:1: val_loss must be finite"),
        ('{"step": 1, "train_loss": 1.0, "lr": -Infinity}\n', r"train\.jsonl:1: lr must be finite"),
    ],
)
def test_load_reports_line_scoped_validation_errors(tmp_path, text, match):
    path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=match):
        plot_loss._load(path)


def test_load_reports_actual_nonblank_line_number(tmp_path):
    path = _write(
        tmp_path,
        '\n{"step": 1, "train_loss": 2.0}\n\n{"step": 2}\n',
    )

    with pytest.raises(ValueError, match=r"train\.jsonl:4: record missing required keys"):
        plot_loss._load(path)


def test_optional_metrics_may_be_null(tmp_path):
    path = _write(
        tmp_path,
        '{"step": 3, "train_loss": 1.5, "val_loss": null, "lr": null}\n',
    )

    records = plot_loss._load(path)

    assert records == [{"step": 3, "train_loss": 1.5, "val_loss": None, "lr": None}]
