"""Regression coverage for deeply nested safe-checkpoint manifests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.safe_checkpoint as safe_checkpoint
from engine import read_safe_checkpoint


def _write_manifest(path, manifest_bytes):
    np.savez(
        path,
        __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
    )


def _valid_state_node(metadata_value):
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
                    "items": [["nested", metadata_value]],
                },
            ],
        ],
    }


def _manifest_with_metadata_node(node_bytes):
    return (
        b'{"safe_checkpoint_version":1,"state":{"type":"dict","items":['
        b'["format_version",{"type":"int","value":2}],'
        b'["model",{"type":"dict","items":[]}],'
        b'["optimizer",{"type":"none"}],'
        b'["optimizer_type",{"type":"none"}],'
        b'["scheduler",{"type":"none"}],'
        b'["rng_state",{"type":"none"}],'
        b'["step",{"type":"int","value":0}],'
        b'["metadata",{"type":"dict","items":[["nested",'
        + node_bytes
        + b']]}]]}}'
    )


def test_json_parser_recursion_error_is_normalized(tmp_path, monkeypatch):
    def fail_with_recursion(*args, **kwargs):
        raise RecursionError("test parser recursion")

    monkeypatch.setattr(safe_checkpoint.json, "loads", fail_with_recursion)
    path = tmp_path / "parser-recursion.npz"
    _write_manifest(path, b"{}")

    with pytest.raises(ValueError, match="manifest nesting is too deep"):
        read_safe_checkpoint(path)


def test_deep_encoded_manifest_recursion_is_normalized(tmp_path):
    depth = sys.getrecursionlimit() * 2
    nested = (
        b'{"type":"list","items":[' * depth
        + b'{"type":"none"}'
        + b"]}" * depth
    )
    path = tmp_path / "too-deep-encoded.npz"
    _write_manifest(path, _manifest_with_metadata_node(nested))

    with pytest.raises(ValueError, match="manifest nesting is too deep"):
        read_safe_checkpoint(path)


def test_decoder_recursion_error_is_normalized(tmp_path, monkeypatch):
    nested = {"type": "none"}
    for _ in range(sys.getrecursionlimit() * 2):
        nested = {"type": "list", "items": [nested]}
    manifest = {
        "safe_checkpoint_version": 1,
        "state": _valid_state_node(nested),
    }
    monkeypatch.setattr(
        safe_checkpoint.json,
        "loads",
        lambda *args, **kwargs: manifest,
    )

    path = tmp_path / "too-deep-node.npz"
    _write_manifest(path, b"{}")

    with pytest.raises(ValueError, match="manifest nesting is too deep"):
        read_safe_checkpoint(path)


def test_moderately_nested_valid_metadata_still_decodes(tmp_path, monkeypatch):
    nested = {"type": "str", "value": "leaf"}
    for _ in range(25):
        nested = {"type": "list", "items": [nested]}
    manifest = {
        "safe_checkpoint_version": 1,
        "state": _valid_state_node(nested),
    }
    monkeypatch.setattr(
        safe_checkpoint.json,
        "loads",
        lambda *args, **kwargs: manifest,
    )

    path = tmp_path / "nested-valid.npz"
    _write_manifest(path, b"{}")
    state = read_safe_checkpoint(path)

    value = state["metadata"]["nested"]
    for _ in range(25):
        assert isinstance(value, list) and len(value) == 1
        value = value[0]
    assert value == "leaf"
