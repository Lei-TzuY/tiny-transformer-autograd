"""Safe-checkpoint inspection CLI and summary regressions."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.safe_checkpoint import save_safe_checkpoint
from safe_checkpoint_inspect import main, summarize_safe_checkpoint


class _Model:
    def state_dict(self):
        return {
            "z.weight": np.arange(6.0).reshape(2, 3),
            "a.bias": np.array([1.0, 2.0]),
        }


class _Optimizer:
    def state_dict(self):
        return {"lr": 0.01}


class _Scheduler:
    def state_dict(self):
        return {"last_step": 6}


def _write_checkpoint(path):
    save_safe_checkpoint(
        path,
        _Model(),
        optimizer=_Optimizer(),
        scheduler=_Scheduler(),
        step=7,
        metadata={"seed": 3, "arch": "tiny"},
    )


def test_summary_counts_and_orders_model_tensors():
    state = {
        "format_version": 2,
        "step": 4,
        "optimizer": None,
        "optimizer_type": None,
        "scheduler": None,
        "metadata": {"z": 1, "a": 2},
        "model": {
            "z": np.zeros((2, 3)),
            "a": np.zeros((4,)),
        },
    }

    summary = summarize_safe_checkpoint(state)

    assert summary["model"]["tensor_count"] == 2
    assert summary["model"]["scalar_count"] == 10
    assert [item["name"] for item in summary["model"]["tensors"]] == ["a", "z"]
    assert summary["metadata_keys"] == ["a", "z"]


def test_summary_rejects_non_array_model_value_explicitly():
    state = {"model": {"weight": [1.0, 2.0]}, "metadata": {}}

    with pytest.raises(TypeError, match="model state value for weight must be a NumPy array"):
        summarize_safe_checkpoint(state)


def test_json_cli_reports_real_safe_checkpoint(tmp_path, capsys):
    path = tmp_path / "model.safe.npz"
    _write_checkpoint(path)

    assert main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["format_version"] == 2
    assert payload["step"] == 7
    assert payload["optimizer_type"] == "_Optimizer"
    assert payload["has_optimizer"] is True
    assert payload["has_scheduler"] is True
    assert payload["metadata_keys"] == ["arch", "seed"]
    assert payload["model"]["tensor_count"] == 2
    assert payload["model"]["scalar_count"] == 8
    assert [item["name"] for item in payload["model"]["tensors"]] == [
        "a.bias",
        "z.weight",
    ]


def test_human_cli_lists_tensors_only_when_requested(tmp_path, capsys):
    path = tmp_path / "model.safe.npz"
    _write_checkpoint(path)

    main([str(path)])
    compact = capsys.readouterr().out
    assert "model_tensors: 2" in compact
    assert "model_scalars: 8" in compact
    assert "tensor a.bias:" not in compact

    main([str(path), "--tensors"])
    detailed = capsys.readouterr().out
    assert "tensor a.bias: shape=2, dtype=float64, scalars=2" in detailed
    assert "tensor z.weight: shape=2x3, dtype=float64, scalars=6" in detailed


def test_invalid_container_is_reported_as_cli_usage_error(tmp_path, capsys):
    path = tmp_path / "not-a-checkpoint.safe.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(SystemExit) as excinfo:
        main([str(path)])

    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    assert "invalid safe checkpoint container" in error
