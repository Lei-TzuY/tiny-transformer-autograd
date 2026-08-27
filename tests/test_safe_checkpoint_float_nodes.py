"""Regression tests for safe-checkpoint JSON float-node decoding."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import read_safe_checkpoint


def _checkpoint_state_node(float_value):
    return {
        "type": "dict",
        "items": [
            ["format_version", {"type": "int", "value": 2}],
            ["model", {"type": "dict", "items": []}],
            ["optimizer", {"type": "none"}],
            ["optimizer_type", {"type": "none"}],
            ["scheduler", {"type": "none"}],
            ["rng_state", {"type": "none"}],
            ["step", {"type": "int", "value": 0}],
            [
                "metadata",
                {
                    "type": "dict",
                    "items": [
                        ["probe", {"type": "float", "value": float_value}],
                    ],
                },
            ],
        ],
    }


def _write_manifest_checkpoint(path, float_value):
    manifest = {
        "safe_checkpoint_version": 1,
        "state": _checkpoint_state_node(float_value),
    }
    payload = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        np.savez(handle, __manifest__=np.frombuffer(payload, dtype=np.uint8))


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_reader_rejects_integer_float_node_that_cannot_fit_float64(tmp_path, value):
    path = tmp_path / "unrepresentable-float.safe.npz"
    _write_manifest_checkpoint(path, value)

    with pytest.raises(
        ValueError,
        match=r"invalid finite float at state\.metadata\.probe",
    ):
        read_safe_checkpoint(path)


def test_reader_accepts_large_integer_float_node_when_float64_representable(tmp_path):
    path = tmp_path / "large-finite-float.safe.npz"
    value = 10**300
    _write_manifest_checkpoint(path, value)

    state = read_safe_checkpoint(path)

    assert type(state["metadata"]["probe"]) is float
    assert state["metadata"]["probe"] == float(value)


def test_reader_preserves_ordinary_finite_float_node(tmp_path):
    path = tmp_path / "ordinary-float.safe.npz"
    _write_manifest_checkpoint(path, -1.25)

    state = read_safe_checkpoint(path)

    assert state["metadata"]["probe"] == -1.25
