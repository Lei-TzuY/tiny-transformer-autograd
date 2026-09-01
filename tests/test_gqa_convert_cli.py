import json

import numpy as np
import pytest

from engine.checkpoint import read_checkpoint, save_checkpoint
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint
from gqa_convert_cli import main
from nn import GPT


def _model(*, kv_heads=None):
    return GPT(
        vocab_size=16,
        context_len=4,
        d_model=8,
        num_heads=4,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        num_kv_heads=kv_heads,
    )


def _metadata(model):
    return {"model_config": model.config()}


def test_cli_auto_converts_pickle_to_safe_and_reports_json(tmp_path, capsys):
    source = tmp_path / "source.pkl"
    destination = tmp_path / "converted.npz"
    model = _model()
    save_checkpoint(source, model, step=7, metadata=_metadata(model))

    report = main(
        [
            str(source),
            str(destination),
            "--kv-heads",
            "2",
            "--json",
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert stdout == report
    assert report == {
        "source_format": "pickle",
        "destination_format": "safe",
        "in_place": False,
        "checkpoint_step": 7,
        "optimizer_type": None,
        "query_heads": 4,
        "kv_heads": 2,
        "kv_cache_head_ratio": 0.5,
    }
    converted = read_safe_checkpoint(destination)
    assert converted["metadata"]["model_config"]["num_kv_heads"] == 2


def test_cli_auto_supports_safe_in_place_mqa_conversion(tmp_path, capsys):
    path = tmp_path / "model.npz"
    model = _model(kv_heads=2)
    save_safe_checkpoint(path, model, step=3, metadata=_metadata(model))

    report = main([str(path), "--kv-heads", "1"])
    output = capsys.readouterr().out

    assert report["source_format"] == "safe"
    assert report["destination_format"] == "safe"
    assert report["in_place"] is True
    assert report["kv_heads"] == 1
    assert "kv_heads=1" in output
    converted = read_safe_checkpoint(path)
    assert converted["metadata"]["model_config"]["num_kv_heads"] == 1


def test_cli_explicit_formats_allow_unknown_extensions(tmp_path):
    source = tmp_path / "source.data"
    destination = tmp_path / "destination.data"
    model = _model()
    save_checkpoint(source, model, metadata=_metadata(model))

    report = main(
        [
            str(source),
            str(destination),
            "--kv-heads",
            "2",
            "--source-format",
            "pickle",
            "--destination-format",
            "pickle",
            "--json",
        ]
    )

    assert report["source_format"] == "pickle"
    assert report["destination_format"] == "pickle"
    converted = read_checkpoint(destination)
    assert converted["metadata"]["model_config"]["num_kv_heads"] == 2


def test_cli_unknown_auto_extension_fails_before_creating_destination(tmp_path):
    source = tmp_path / "source.unknown"
    destination = tmp_path / "destination.pkl"
    source.write_bytes(b"not opened")

    with pytest.raises(SystemExit) as excinfo:
        main([str(source), str(destination), "--kv-heads", "2"])

    assert excinfo.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("value", ["0", "-1", "nope"])
def test_cli_rejects_invalid_kv_head_argument(value):
    with pytest.raises(SystemExit) as excinfo:
        main(["input.pkl", "output.pkl", "--kv-heads", value])
    assert excinfo.value.code == 2


def test_cli_report_is_strict_json_safe(tmp_path):
    source = tmp_path / "source.pkl"
    destination = tmp_path / "destination.pkl"
    model = _model()
    save_checkpoint(source, model, metadata=_metadata(model))

    report = main(
        [
            str(source),
            str(destination),
            "--kv-heads",
            "4",
            "--json",
        ]
    )
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_cli_conversion_does_not_perturb_process_rng(tmp_path):
    source = tmp_path / "source.pkl"
    destination = tmp_path / "destination.pkl"
    model = _model()
    save_checkpoint(source, model, metadata=_metadata(model))

    np.random.seed(12345)
    before = np.random.get_state()
    main([str(source), str(destination), "--kv-heads", "2", "--json"])
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
