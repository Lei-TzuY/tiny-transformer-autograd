"""Installed console-script smoke tests for GQA-aware training entry points."""

import shutil
import subprocess

import pytest


@pytest.mark.parametrize("command", ["tiny-train", "tiny-train-safe"])
def test_installed_training_help_lists_kv_heads(command):
    executable = shutil.which(command)
    assert executable is not None, f"installed console script {command!r} was not found"
    completed = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--kv-heads" in completed.stdout
    assert "usage:" in completed.stdout.lower()
    assert completed.stderr == ""
