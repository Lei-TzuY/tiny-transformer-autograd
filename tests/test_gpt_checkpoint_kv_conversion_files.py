import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import read_checkpoint, save_checkpoint
from engine.optim import Adam
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint
from engine.scheduler import WarmupCosineScheduler
from nn import GPT, convert_gpt_checkpoint_file


def _model():
    np.random.seed(31)
    return GPT(
        vocab_size=17,
        context_len=4,
        d_model=8,
        num_heads=4,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
    )


def _training_objects(model):
    optimizer = Adam(
        model.parameters(),
        lr=0.004,
        betas=(0.75, 0.9),
        eps=2e-7,
        weight_decay=0.02,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=20, warmup_steps=2, min_lr=0.0004
    )
    scheduler.step(3)
    for parameter in model.parameters():
        parameter.grad = np.full_like(parameter.data, 0.125)
    optimizer.step()
    return optimizer, scheduler


def _metadata(model, *, extra=None):
    result = {
        "model_config": model.config(),
        "tokenizer": {"kind": "char", "vocab": ["a", "b"]},
    }
    if extra is not None:
        result["extra"] = extra
    return result


def _load_model(state):
    model = GPT(**state["metadata"]["model_config"])
    model.load_state_dict(state["model"], strict=True)
    return model


def test_pickle_checkpoint_converts_to_safe_file(tmp_path):
    source = _model()
    optimizer, scheduler = _training_objects(source)
    source_path = tmp_path / "source.pkl"
    target_path = tmp_path / "converted.npz"
    save_checkpoint(
        source_path,
        source,
        optimizer,
        scheduler,
        step=6,
        metadata=_metadata(source),
    )

    returned = convert_gpt_checkpoint_file(
        source_path,
        target_path,
        2,
        source_format="pickle",
        destination_format="safe",
    )
    loaded = read_safe_checkpoint(target_path)

    assert loaded["metadata"]["model_config"]["num_kv_heads"] == 2
    assert loaded["step"] == 6
    assert loaded["optimizer_type"] == "Adam"
    assert loaded["optimizer"]["t"] == 0
    assert loaded["scheduler"] == scheduler.state_dict()
    assert returned["metadata"]["model_config"] == loaded["metadata"]["model_config"]
    _load_model(loaded)


def test_safe_checkpoint_can_be_converted_in_place(tmp_path):
    source = _model()
    optimizer, scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.npz"
    save_safe_checkpoint(
        path,
        source,
        optimizer,
        scheduler,
        step=8,
        metadata=_metadata(source),
    )

    convert_gpt_checkpoint_file(
        path,
        path,
        1,
        source_format="safe",
    )
    loaded = read_safe_checkpoint(path)

    assert loaded["metadata"]["model_config"]["num_kv_heads"] == 1
    assert loaded["optimizer"]["t"] == 0
    assert loaded["step"] == 8
    _load_model(loaded)


def test_safe_checkpoint_converts_to_pickle_file(tmp_path):
    source = _model()
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "converted.pkl"
    save_safe_checkpoint(
        source_path,
        source,
        step=3,
        metadata=_metadata(source),
    )

    convert_gpt_checkpoint_file(
        source_path,
        target_path,
        2,
        source_format="safe",
        destination_format="pickle",
    )
    loaded = read_checkpoint(target_path)

    assert loaded["metadata"]["model_config"]["num_kv_heads"] == 2
    assert loaded["optimizer"] is None
    assert loaded["step"] == 3
    _load_model(loaded)


def test_failed_safe_encoding_does_not_replace_existing_destination(tmp_path):
    source = _model()
    source_path = tmp_path / "source.pkl"
    target_path = tmp_path / "destination.npz"
    target_path.write_bytes(b"sentinel")
    save_checkpoint(
        source_path,
        source,
        step=0,
        metadata=_metadata(source, extra={1, 2, 3}),
    )

    with pytest.raises(TypeError, match="safe checkpoint does not support"):
        convert_gpt_checkpoint_file(
            source_path,
            target_path,
            2,
            source_format="pickle",
            destination_format="safe",
        )

    assert target_path.read_bytes() == b"sentinel"


@pytest.mark.parametrize("name", ["source_format", "destination_format"])
def test_checkpoint_file_format_validation(name, tmp_path):
    source_path = tmp_path / "source"
    target_path = tmp_path / "target"
    kwargs = {"source_format": "pickle"}
    kwargs[name] = "tar"

    with pytest.raises(ValueError, match="must be 'pickle' or 'safe'"):
        convert_gpt_checkpoint_file(source_path, target_path, 2, **kwargs)


def test_checkpoint_file_format_rejects_non_strings_before_io(tmp_path):
    with pytest.raises(TypeError, match="source_format"):
        convert_gpt_checkpoint_file(
            tmp_path / "missing",
            tmp_path / "target",
            2,
            source_format=1,
        )
