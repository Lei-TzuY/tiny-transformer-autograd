"""Crash-safety regression tests for trusted pickle checkpoint saves."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.checkpoint as checkpoint


class _StateModel:
    def __init__(self, value=1.0):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}


class _SerializationBomb:
    def __reduce__(self):
        raise RuntimeError("serialization boom")


class _UnpicklableModel:
    def state_dict(self):
        return {"bad": _SerializationBomb()}


def _temporary_files(path):
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_success_fsyncs_temp_before_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "model.ckpt"
    events = []
    real_replace = checkpoint.os.replace

    def fake_fsync(descriptor):
        events.append(("fsync", descriptor))

    def checked_replace(source, destination):
        events.append(("replace", source))
        assert any(kind == "fsync" for kind, _ in events)
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "fsync", fake_fsync)
    monkeypatch.setattr(checkpoint.os, "replace", checked_replace)

    checkpoint.save_checkpoint(path, _StateModel(3.5), step=4)

    assert [kind for kind, _ in events] == ["fsync", "replace"]
    assert not _temporary_files(path)
    state = checkpoint.read_checkpoint(path)
    assert state["step"] == 4
    np.testing.assert_array_equal(state["model"]["value"], [3.5])


def test_serialization_failure_preserves_destination_and_cleans_temp(tmp_path):
    path = tmp_path / "model.ckpt"
    original = b"existing checkpoint bytes"
    path.write_bytes(original)

    with pytest.raises(RuntimeError, match="serialization boom"):
        checkpoint.save_checkpoint(path, _UnpicklableModel())

    assert path.read_bytes() == original
    assert not _temporary_files(path)


def test_fsync_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "model.ckpt"
    original = b"existing checkpoint bytes"
    path.write_bytes(original)

    def fail_fsync(descriptor):
        raise OSError("fsync boom")

    monkeypatch.setattr(checkpoint.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync boom"):
        checkpoint.save_checkpoint(path, _StateModel(2.0))

    assert path.read_bytes() == original
    assert not _temporary_files(path)


def test_replace_failure_preserves_destination_and_uses_unique_temps(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.ckpt"
    original = b"existing checkpoint bytes"
    path.write_bytes(original)
    sources = []

    def fail_replace(source, destination):
        sources.append(source)
        assert os.fspath(destination) == os.fspath(path)
        raise OSError("replace boom")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)

    for value in (1.0, 2.0):
        with pytest.raises(OSError, match="replace boom"):
            checkpoint.save_checkpoint(path, _StateModel(value))
        assert path.read_bytes() == original
        assert not _temporary_files(path)

    assert len(sources) == 2
    assert sources[0] != sources[1]
    for source in sources:
        assert os.path.dirname(source) == os.fspath(tmp_path)
        assert os.path.basename(source).startswith(f".{path.name}.")
        assert source.endswith(".tmp")
