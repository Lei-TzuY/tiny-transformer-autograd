import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.checkpoint import CHECKPOINT_VERSION
from engine.optim import Adam
from nn import GPT, convert_gpt_checkpoint_kv_heads


def _model(*, heads=4, kv_heads=None, d_model=8):
    np.random.seed(41)
    kwargs = {}
    if kv_heads is not None:
        kwargs["num_kv_heads"] = kv_heads
    return GPT(
        vocab_size=13,
        context_len=4,
        d_model=d_model,
        num_heads=heads,
        d_ff=2 * d_model,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    )


def _checkpoint(model, *, optimizer=None, optimizer_type=None, metadata=None):
    return {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "optimizer_type": optimizer_type,
        "scheduler": None,
        "rng_state": np.random.get_state(),
        "step": 4,
        "metadata": (
            {"model_config": model.config()}
            if metadata is None
            else metadata
        ),
    }


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_checkpoint_must_be_mapping():
    with pytest.raises(TypeError, match="checkpoint must be a mapping"):
        convert_gpt_checkpoint_kv_heads([], 2)


def test_checkpoint_requires_model_config_mapping():
    source = _model()
    missing = _checkpoint(source, metadata={})
    malformed = _checkpoint(source, metadata={"model_config": []})

    with pytest.raises(ValueError, match="model_config mapping"):
        convert_gpt_checkpoint_kv_heads(missing, 2)
    with pytest.raises(ValueError, match="model_config mapping"):
        convert_gpt_checkpoint_kv_heads(malformed, 2)


def test_checkpoint_model_state_must_match_declared_config():
    source = _model()
    checkpoint = _checkpoint(source)
    checkpoint["metadata"]["model_config"] = dict(checkpoint["metadata"]["model_config"])
    checkpoint["metadata"]["model_config"]["d_model"] = 12

    with pytest.raises(ValueError, match="shape mismatch"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 2)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, "2", None])
def test_target_kv_heads_reuses_strict_public_integer_validation(bad):
    with pytest.raises(TypeError, match="positive integer"):
        convert_gpt_checkpoint_kv_heads(_checkpoint(_model()), bad)


def test_crossing_kv_partitions_are_rejected():
    source = _model(heads=12, kv_heads=3, d_model=24)
    with pytest.raises(ValueError, match="divide one another"):
        convert_gpt_checkpoint_kv_heads(_checkpoint(source), 4)


def test_unknown_optimizer_is_rejected_only_when_reparameterization_needs_reset():
    source = _model()
    checkpoint = _checkpoint(source)
    checkpoint["optimizer"] = {"opaque": np.array([1.0])}
    checkpoint["optimizer_type"] = "CustomOptimizer"

    same = convert_gpt_checkpoint_kv_heads(checkpoint, 4)
    assert same["optimizer_type"] == "CustomOptimizer"
    np.testing.assert_array_equal(same["optimizer"]["opaque"], np.array([1.0]))

    with pytest.raises(ValueError, match="unsupported checkpoint optimizer type"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 2)


def test_malformed_builtin_optimizer_state_fails_before_output_and_restores_rng():
    source = _model()
    optimizer = Adam(source.parameters(), lr=0.01)
    checkpoint = _checkpoint(source, optimizer=optimizer, optimizer_type="Adam")
    checkpoint["optimizer"]["m"][-1] = np.zeros((99,), dtype=np.float64)

    np.random.seed(919191)
    rng_before = np.random.get_state()
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 2)
    _assert_rng_equal(np.random.get_state(), rng_before)


def test_reserved_migration_history_must_be_list_and_input_is_unchanged():
    source = _model()
    checkpoint = _checkpoint(
        source,
        metadata={
            "model_config": source.config(),
            "_tiny_transformer_migrations": "not-a-list",
        },
    )
    state_before = {name: value.copy() for name, value in checkpoint["model"].items()}

    with pytest.raises(TypeError, match="must be a list"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    assert checkpoint["metadata"]["_tiny_transformer_migrations"] == "not-a-list"
    for name, expected in state_before.items():
        np.testing.assert_array_equal(checkpoint["model"][name], expected)


def test_existing_migration_history_is_appended_without_aliasing():
    source = _model()
    history = [
        {
            "kind": "earlier",
            "source_num_kv_heads": 4,
            "target_num_kv_heads": 4,
            "optimizer_state": "preserved",
        }
    ]
    checkpoint = _checkpoint(
        source,
        metadata={
            "model_config": source.config(),
            "_tiny_transformer_migrations": history,
        },
    )

    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    assert len(converted["metadata"]["_tiny_transformer_migrations"]) == 2
    assert converted["metadata"]["_tiny_transformer_migrations"][0] == history[0]
    converted["metadata"]["_tiny_transformer_migrations"][0]["kind"] = "changed"
    assert history[0]["kind"] == "earlier"


def test_cyclic_checkpoint_containers_are_rejected_without_recursion_error():
    source = _model()
    checkpoint = _checkpoint(source)
    cycle = []
    cycle.append(cycle)
    checkpoint["metadata"]["cycle"] = cycle

    with pytest.raises(ValueError, match="cyclic list"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 2)


def test_output_arrays_are_detached_from_input_checkpoint():
    source = _model()
    checkpoint = _checkpoint(source)
    converted = convert_gpt_checkpoint_kv_heads(checkpoint, 2)

    for name, source_value in checkpoint["model"].items():
        converted_value = converted["model"][name]
        if converted_value.shape == source_value.shape:
            assert not np.shares_memory(converted_value, source_value)


def test_global_rng_is_restored_when_model_config_constructor_fails():
    source = _model()
    checkpoint = _checkpoint(source)
    checkpoint["metadata"]["model_config"] = dict(checkpoint["metadata"]["model_config"])
    checkpoint["metadata"]["model_config"]["num_heads"] = 3

    np.random.seed(123456)
    rng_before = np.random.get_state()
    with pytest.raises(ValueError, match="divisible"):
        convert_gpt_checkpoint_kv_heads(checkpoint, 1)
    _assert_rng_equal(np.random.get_state(), rng_before)
