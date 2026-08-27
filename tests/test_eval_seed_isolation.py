"""Regression tests for deterministic validation RNG isolation."""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train


class _MetadataModel:
    def config(self):
        return {"vocab_size": 3, "context_len": 2}


class _MetadataTokenizer:
    def state_dict(self):
        return {"kind": "char", "vocab": ["a", "b", "c"]}


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _training_trace(output):
    pattern = re.compile(
        r"step\s+(\d+)/\d+\s+train_loss=([0-9.]+).*?gnorm=([0-9.]+)"
    )
    return [
        (int(step), float(loss), float(grad_norm))
        for step, loss, grad_norm in pattern.findall(output)
    ]


def _run_cli(monkeypatch, capsys, *, eval_iters, eval_seed):
    argv = [
        "tiny-train",
        "--iters", "3",
        "--eval-interval", "1",
        "--eval-iters", str(eval_iters),
        "--ctx", "4",
        "--d", "4",
        "--heads", "1",
        "--layers", "1",
        "--batch", "2",
        "--dropout", "0.2",
        "--seed", "17",
        "--no-sample",
    ]
    if eval_seed is not None:
        argv.extend(["--eval-seed", str(eval_seed)])
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    return capsys.readouterr().out


def test_seeded_evaluation_repeats_samples_and_restores_training_rng():
    np.random.seed(1234)
    before = np.random.get_state()

    def sample_validation():
        return tuple(np.random.randint(0, 1000, size=8).tolist())

    first = train._run_eval_with_seed(sample_validation, 77)
    after_first = np.random.get_state()
    second = train._run_eval_with_seed(sample_validation, 77)
    after_second = np.random.get_state()

    assert first == second
    _assert_rng_state_equal(after_first, before)
    _assert_rng_state_equal(after_second, before)


def test_seeded_evaluation_restores_rng_when_validation_raises():
    np.random.seed(456)
    before = np.random.get_state()

    def fail_after_draw():
        np.random.random(20)
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        train._run_eval_with_seed(fail_after_draw, 8)

    _assert_rng_state_equal(np.random.get_state(), before)


def test_unset_eval_seed_preserves_historical_rng_consumption():
    np.random.seed(91)
    expected_first = int(np.random.randint(0, 100000))
    expected_second = int(np.random.randint(0, 100000))

    np.random.seed(91)
    observed_first = train._run_eval_with_seed(
        lambda: int(np.random.randint(0, 100000)), None
    )
    observed_second = int(np.random.randint(0, 100000))

    assert observed_first == expected_first
    assert observed_second == expected_second


def test_eval_seed_isolation_makes_training_independent_of_eval_iters(
    monkeypatch, capsys
):
    short_eval = _training_trace(
        _run_cli(monkeypatch, capsys, eval_iters=1, eval_seed=123)
    )
    long_eval = _training_trace(
        _run_cli(monkeypatch, capsys, eval_iters=4, eval_seed=123)
    )

    assert len(short_eval) == len(long_eval) == 3
    assert short_eval == long_eval


def test_legacy_evaluation_rng_coupling_is_preserved_without_eval_seed(
    monkeypatch, capsys
):
    short_eval = _training_trace(
        _run_cli(monkeypatch, capsys, eval_iters=1, eval_seed=None)
    )
    long_eval = _training_trace(
        _run_cli(monkeypatch, capsys, eval_iters=4, eval_seed=None)
    )

    assert short_eval[0] == long_eval[0]
    assert short_eval[1:] != long_eval[1:]


def test_parse_args_distinguishes_omitted_and_explicit_eval_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiny-train"])
    assert train.parse_args().eval_seed is None

    monkeypatch.setattr(sys, "argv", ["tiny-train", "--eval-seed", "123"])
    assert train.parse_args().eval_seed == 123


def test_validate_args_rejects_invalid_eval_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiny-train", "--eval-seed", "-1"])
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        train._validate_args(train.parse_args())

    monkeypatch.setattr(
        sys, "argv", ["tiny-train", "--eval-seed", str(2**32)]
    )
    with pytest.raises(ValueError, match=r"\[0, 4294967295\]"):
        train._validate_args(train.parse_args())


def test_normalise_eval_seed_rejects_bool_and_accepts_numpy_integer():
    with pytest.raises(TypeError, match="integer"):
        train._normalise_eval_seed(True)
    assert train._normalise_eval_seed(np.int64(99)) == 99


def test_metadata_records_eval_seed_without_forcing_label_smoothing():
    metadata = train._metadata(
        _MetadataModel(), _MetadataTokenizer(), label_smoothing=0.0, eval_seed=41
    )

    assert metadata["training_config"] == {"eval_seed": 41}


def test_metadata_combines_training_objective_and_eval_seed():
    metadata = train._metadata(
        _MetadataModel(), _MetadataTokenizer(), label_smoothing=0.2, eval_seed=41
    )

    assert metadata["training_config"] == {
        "label_smoothing": 0.2,
        "eval_seed": 41,
    }


def test_eval_seed_resolution_defaults_inherits_and_matches_checkpoint():
    assert train._resolve_eval_seed(None, {}) is None
    assert train._resolve_eval_seed(17, {}) == 17

    metadata = {"training_config": {"eval_seed": 123}}
    assert train._resolve_eval_seed(None, metadata) == 123
    assert train._resolve_eval_seed(123, metadata) == 123


def test_eval_seed_resolution_rejects_rng_semantic_change_on_resume():
    metadata = {"training_config": {"eval_seed": 123}}

    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        train._resolve_eval_seed(456, metadata)


@pytest.mark.parametrize(
    "metadata, error, message",
    [
        ({"training_config": []}, ValueError, "must be a mapping"),
        ({"training_config": {"eval_seed": True}}, TypeError, "integer"),
        ({"training_config": {"eval_seed": -1}}, ValueError, r"\[0, 4294967295\]"),
        (
            {"training_config": {"eval_seed": 2**32}},
            ValueError,
            r"\[0, 4294967295\]",
        ),
    ],
)
def test_eval_seed_resolution_validates_checkpoint_metadata(metadata, error, message):
    with pytest.raises(error, match=message):
        train._resolve_eval_seed(None, metadata)


def test_metadata_and_resolution_round_trip_eval_seed():
    metadata = train._metadata(
        _MetadataModel(), _MetadataTokenizer(), label_smoothing=0.3, eval_seed=19
    )

    assert train._resolve_eval_seed(None, metadata) == 19
    assert train._resolve_label_smoothing(None, metadata) == 0.3
