"""Strict JSON parsing regressions for safe checkpoints."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine import read_safe_checkpoint


def test_reader_rejects_duplicate_json_object_keys(tmp_path):
    path = tmp_path / "duplicate-json-key.safe.npz"
    # RFC 8259 permits parsers to choose how duplicate object names are handled.
    # A security boundary must reject the ambiguity rather than silently letting
    # the later value replace the earlier one.
    manifest_bytes = (
        b'{"safe_checkpoint_version":1,'
        b'"state":{"type":"execute","type":"dict","items":[]}}'
    )
    with path.open("wb") as handle:
        np.savez(
            handle,
            __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        read_safe_checkpoint(path)
