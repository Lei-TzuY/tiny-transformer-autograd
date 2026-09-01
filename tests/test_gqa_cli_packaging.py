import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "command",
    ["tiny-gqa-convert", "tiny-gqa-benchmark"],
)
def test_installed_gqa_console_script_help(command):
    executable = shutil.which(command)
    assert executable is not None, f"installed console script {command!r} was not found"

    completed = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower()
    assert completed.stderr == ""
