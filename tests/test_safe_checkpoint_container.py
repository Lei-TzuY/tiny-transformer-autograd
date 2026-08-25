"""Container-boundary tests for safe checkpoints."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import read_safe_checkpoint


def test_plain_npy_file_is_rejected_as_the_wrong_container_type(tmp_path):
    path = tmp_path / "not-an-archive.npy"
    np.save(path, np.arange(4, dtype=np.float64))

    with pytest.raises(ValueError, match="must be an NPZ archive"):
        read_safe_checkpoint(path)
