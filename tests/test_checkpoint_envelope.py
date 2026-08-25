"""Regression tests for checkpoint envelope validation."""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


class _StateModel:
    def __init__(self, value=1.0):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value.copy()}

    def load_state_dict(self, state, strict=True):
        self.value[:] = state["value"]


class _SnapshotBomb:
    def state_dict(self):
        raise AssertionError("caller state must not be snapshotted")

    def load_state_dict(self, state, strict=True):
        raise AssertionError("caller state must not be mutated")


def _assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _write_trusted_pickle(path, state):
    with open(path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("state", [None, [], (), "checkpoint"])
def test_restore_requires_mapping_before_touching_model(state):
    with pytest.raises(TypeError, match="mapping"):
        restore_checkpoint(state, _SnapshotBomb())


@pytest.mark.parametrize("bad_version", [True, np.bool_(False), 1.5, "2", None])
def test_restore_rejects_non_integral_format_version_before_snapshot(bad_version):
    state = {"format_version": bad_version, "model": {}}
    with pytest.raises(TypeError, match="format_version"):
        restore_checkpoint(state, _SnapshotBomb())


@pytest.mark.parametrize("bad_version", [0, -1, 3, 999])
def test_restore_rejects_unsupported_integral_format_version_before_snapshot(
    bad_version,
):
    state = {"format_version": bad_version, "model": {}}
    with pytest.raises(ValueError, match="format version"):
        restore_checkpoint(state, _SnapshotBomb())


def test_numpy_integral_format_version_remains_supported():
    source = _StateModel(4.0)
    target = _StateModel(9.0)
    state = {
        "format_version": np.int64(2),
        "model": source.state_dict(),
        "step": np.int64(7),
    }

    assert restore_checkpoint(state, target) == 7
    np.testing.assert_array_equal(target.value, source.value)


def test_restore_requires_model_section_before_snapshot():
    with pytest.raises(ValueError, match="model state"):
        restore_checkpoint({"format_version": 2}, _SnapshotBomb())


@pytest.mark.parametrize("bad_step", [True, np.bool_(False), 1.5, "3", None, np.nan])
def test_restore_rejects_non_integral_step_before_snapshot(bad_step):
    state = {"format_version": 2, "model": {}, "step": bad_step}
    with pytest.raises(TypeError, match="non-negative integer"):
        restore_checkpoint(state, _SnapshotBomb())


def test_restore_rejects_negative_step_before_snapshot():
    state = {"format_version": 2, "model": {}, "step": -1}
    with pytest.raises(ValueError, match="non-negative integer"):
        restore_checkpoint(state, _SnapshotBomb())


@pytest.mark.parametrize("bad_optimizer_type", [1, False, "", [], {}])
def test_restore_rejects_malformed_optimizer_type_even_without_optimizer(
    bad_optimizer_type,
):
    state = {
        "format_version": 2,
        "model": {},
        "optimizer_type": bad_optimizer_type,
    }
    with pytest.raises(TypeError, match="optimizer_type"):
        restore_checkpoint(state, _SnapshotBomb())


@pytest.mark.parametrize("bad_strict", [None, 0, 1, "yes", np.bool_(True)])
def test_restore_requires_boolean_strict_before_snapshot(bad_strict):
    state = {"format_version": 2, "model": {}}
    with pytest.raises(TypeError, match="strict"):
        restore_checkpoint(state, _SnapshotBomb(), strict=bad_strict)


def test_malformed_rng_state_is_rejected_without_touching_global_rng_or_model():
    np.random.seed(12345)
    before = np.random.get_state()
    bad_rng = ("not-a-bit-generator",) + tuple(before[1:])
    state = {"format_version": 2, "model": {}, "rng_state": bad_rng}

    with pytest.raises(ValueError, match="RNG state"):
        restore_checkpoint(state, _SnapshotBomb())

    _assert_rng_equal(np.random.get_state(), before)


def test_legacy_missing_version_step_rng_and_metadata_still_restore():
    source = _StateModel(5.0)
    target = _StateModel(-1.0)
    state = {"model": source.state_dict()}

    assert restore_checkpoint(state, target) == 0
    np.testing.assert_array_equal(target.value, source.value)


@pytest.mark.parametrize("bad_step", [True, 1.5, "2", -1])
def test_save_rejects_invalid_step_before_reading_model_state(tmp_path, bad_step):
    path = tmp_path / "invalid.ckpt"
    error = TypeError if bad_step != -1 else ValueError

    with pytest.raises(error, match="non-negative integer"):
        save_checkpoint(path, _SnapshotBomb(), step=bad_step)

    assert not path.exists()


@pytest.mark.parametrize("bad_metadata", [[], (), "metadata", 3, False])
def test_save_rejects_non_mapping_metadata_before_reading_model_state(
    tmp_path, bad_metadata
):
    path = tmp_path / "invalid-metadata.ckpt"

    with pytest.raises(TypeError, match="metadata"):
        save_checkpoint(path, _SnapshotBomb(), metadata=bad_metadata)

    assert not path.exists()


def test_save_normalizes_numpy_integral_step(tmp_path):
    path = tmp_path / "valid.ckpt"
    model = _StateModel(3.0)

    save_checkpoint(path, model, step=np.int64(6))
    state = read_checkpoint(path)

    assert state["step"] == 6
    assert type(state["step"]) is int


def test_read_rejects_malformed_envelope_before_caller_can_use_metadata(tmp_path):
    path = tmp_path / "bad-envelope.ckpt"
    _write_trusted_pickle(
        path,
        {
            "format_version": 2,
            "model": {"value": np.array([1.0])},
            "step": "not-an-integer",
            "metadata": {},
        },
    )

    with pytest.raises(TypeError, match="non-negative integer"):
        read_checkpoint(path)


def test_read_rejects_non_mapping_metadata(tmp_path):
    path = tmp_path / "bad-metadata.ckpt"
    _write_trusted_pickle(
        path,
        {
            "format_version": 2,
            "model": {"value": np.array([1.0])},
            "metadata": [],
        },
    )

    with pytest.raises(TypeError, match="metadata"):
        read_checkpoint(path)


def test_read_preserves_legacy_checkpoint_without_metadata(tmp_path):
    path = tmp_path / "legacy.ckpt"
    state = {"model": {"value": np.array([2.0])}}
    _write_trusted_pickle(path, state)

    loaded = read_checkpoint(path)

    np.testing.assert_array_equal(loaded["model"]["value"], state["model"]["value"])
    assert "metadata" not in loaded
