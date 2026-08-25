"""Tests for the non-pickle NumPy/JSON checkpoint format."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import (
    Adam,
    WarmupCosineScheduler,
    read_safe_checkpoint,
    restore_checkpoint,
    save_safe_checkpoint,
)
from nn.layers import Linear


def _training_objects():
    np.random.seed(41)
    model = Linear(3, 2)
    optimizer = Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=8,
        warmup_steps=2,
        min_lr=2e-4,
    )
    for index, parameter in enumerate(model.parameters()):
        parameter.grad[:] = np.arange(parameter.data.size).reshape(
            parameter.shape
        ) + 0.25 + index
    scheduler.step(2)
    optimizer.step()
    return model, optimizer, scheduler


def _assert_nested_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert list(actual) == list(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
        return
    assert type(actual) is type(expected)
    assert actual == expected


def _rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_safe_checkpoint_roundtrips_full_training_state(tmp_path):
    model, optimizer, scheduler = _training_objects()
    metadata = {
        "model_config": {"width": 3, "dropout": 0.0, "tied": False},
        "tokenizer": {
            "kind": "bpe",
            "vocab": ["a", "β", "ab"],
            "merges": [["a", "b"]],
        },
        "tuple_marker": (1, "two", True),
    }
    path = tmp_path / "training.safe.npz"

    np.random.seed(1234)
    expected_rng = np.random.get_state()
    save_safe_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        step=7,
        metadata=metadata,
    )
    state = read_safe_checkpoint(path)

    assert state["format_version"] == 2
    assert state["optimizer_type"] == "Adam"
    assert state["step"] == 7
    _assert_nested_equal(state["model"], model.state_dict())
    _assert_nested_equal(state["optimizer"], optimizer.state_dict())
    _assert_nested_equal(state["scheduler"], scheduler.state_dict())
    _assert_nested_equal(state["metadata"], metadata)
    assert _rng_equal(state["rng_state"], expected_rng)


def test_safe_checkpoint_state_restores_with_existing_transactional_loader(tmp_path):
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "restore.safe.npz"
    np.random.seed(77)
    save_safe_checkpoint(path, model, optimizer, scheduler, step=5)

    expected_model = model.state_dict()
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    expected_rng = np.random.get_state()

    for parameter in model.parameters():
        parameter.data += 100.0
    optimizer.lr = 9.0
    optimizer.t = 99
    for buffer in optimizer._m + optimizer._v:
        buffer[:] = -17.0
    scheduler.last_step = 7
    scheduler.base_lr = 4.0
    np.random.seed(999)

    step = restore_checkpoint(
        read_safe_checkpoint(path),
        model,
        optimizer,
        scheduler,
    )

    assert step == 5
    _assert_nested_equal(model.state_dict(), expected_model)
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler)
    assert _rng_equal(np.random.get_state(), expected_rng)


def test_failed_safe_save_keeps_existing_destination_unchanged(tmp_path):
    model, _, _ = _training_objects()
    path = tmp_path / "existing.safe.npz"
    original = b"previous checkpoint bytes"
    path.write_bytes(original)

    with pytest.raises(TypeError, match="does not support object"):
        save_safe_checkpoint(path, model, metadata={"unsupported": object()})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_reader_rejects_object_array_even_when_manifest_references_it(tmp_path):
    path = tmp_path / "object.safe.npz"
    manifest = {
        "safe_checkpoint_version": 1,
        "state": {
            "type": "dict",
            "items": [
                [
                    "payload",
                    {"type": "array", "key": "array_00000000"},
                ]
            ],
        },
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        np.savez(
            handle,
            __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
            array_00000000=np.array([{"callable": "never"}], dtype=object),
        )

    with pytest.raises(ValueError, match="require pickle|object array"):
        read_safe_checkpoint(path)


def test_reader_rejects_unmanifested_archive_members(tmp_path):
    path = tmp_path / "extra.safe.npz"
    manifest = {
        "safe_checkpoint_version": 1,
        "state": {"type": "dict", "items": []},
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        np.savez(
            handle,
            __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
            hidden=np.array([1, 2, 3]),
        )

    with pytest.raises(ValueError, match="archive members do not match"):
        read_safe_checkpoint(path)


def test_reader_rejects_manifest_with_unknown_node_type(tmp_path):
    path = tmp_path / "unknown.safe.npz"
    manifest = {
        "safe_checkpoint_version": 1,
        "state": {"type": "execute", "value": "nope"},
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        np.savez(
            handle,
            __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="unknown safe checkpoint node type"):
        read_safe_checkpoint(path)
