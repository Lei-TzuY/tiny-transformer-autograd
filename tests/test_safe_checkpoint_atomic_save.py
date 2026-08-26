"""Crash-durability regression for non-executable safe checkpoints."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.safe_checkpoint as safe_checkpoint


class _StateModel:
    def __init__(self, value=1.0):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}


def test_safe_checkpoint_fsyncs_parent_after_replace(tmp_path, monkeypatch):
    path = tmp_path / "model.safe.npz"
    events = []
    real_replace = safe_checkpoint.os.replace

    def checked_replace(source, destination):
        events.append(("replace", os.fspath(destination)))
        return real_replace(source, destination)

    def checked_directory_fsync(directory):
        events.append(("directory_fsync", os.fspath(directory)))

    monkeypatch.setattr(safe_checkpoint.os, "replace", checked_replace)
    monkeypatch.setattr(
        safe_checkpoint,
        "_fsync_parent_directory",
        checked_directory_fsync,
    )

    safe_checkpoint.save_safe_checkpoint(path, _StateModel(3.5), step=4)

    assert events == [
        ("replace", os.fspath(path)),
        ("directory_fsync", os.fspath(tmp_path)),
    ]
    state = safe_checkpoint.read_safe_checkpoint(path)
    assert state["step"] == 4
    np.testing.assert_array_equal(state["model"]["value"], [3.5])
