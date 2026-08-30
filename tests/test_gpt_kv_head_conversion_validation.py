import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn import GPT, convert_gpt_kv_heads


def _model(*, heads=4, kv_heads=None, d_model=8):
    np.random.seed(7)
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


def _rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, "2", None])
def test_target_kv_heads_requires_integral_value(bad):
    with pytest.raises(TypeError, match="positive integer"):
        convert_gpt_kv_heads(_model(), bad)


@pytest.mark.parametrize("bad", [0, -1, -4])
def test_target_kv_heads_requires_positive_value(bad):
    with pytest.raises(ValueError, match="positive"):
        convert_gpt_kv_heads(_model(), bad)


def test_target_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError, match="divisible"):
        convert_gpt_kv_heads(_model(), 3)


def test_crossing_source_and_target_partitions_are_rejected():
    source = _model(heads=12, kv_heads=3, d_model=24)
    with pytest.raises(ValueError, match="divide one another"):
        convert_gpt_kv_heads(source, 4)


def test_non_gpt_input_is_rejected():
    with pytest.raises(TypeError, match="model must be a GPT"):
        convert_gpt_kv_heads(object(), 1)


def test_numpy_integer_target_is_accepted_and_normalized():
    converted = convert_gpt_kv_heads(_model(), np.int64(2))
    assert converted.num_kv_heads == 2
    assert type(converted.num_kv_heads) is int


def test_nonfinite_source_kv_projection_fails_without_mutation_or_rng_drift():
    source = _model()
    source.blocks[0].attn.W_k.weight.data[0, 0] = np.inf
    state_before = source.state_dict()
    versions_before = {
        name: tensor._version for name, tensor in source.named_tensors()
    }
    np.random.seed(9876)
    rng_before = np.random.get_state()

    with pytest.raises(ValueError, match="only finite"):
        convert_gpt_kv_heads(source, 2)

    _rng_state_equal(np.random.get_state(), rng_before)
    for name, tensor in source.named_tensors():
        np.testing.assert_array_equal(tensor.data, state_before[name])
        assert tensor._version == versions_before[name]


def test_nonfinite_unrelated_source_tensor_fails_transactionally():
    source = _model()
    source.token_emb.weight.data[0, 0] = np.nan
    state_before = source.state_dict()
    np.random.seed(54321)
    rng_before = np.random.get_state()

    with pytest.raises(ValueError, match="only finite"):
        convert_gpt_kv_heads(source, 2)

    _rng_state_equal(np.random.get_state(), rng_before)
    assert np.isnan(source.token_emb.weight.data[0, 0])
    current = dict(source.named_tensors())
    for name, value in state_before.items():
        assert np.array_equal(current[name].data, value, equal_nan=True)


def test_conversion_is_globally_rng_neutral_on_success():
    source = _model(kv_heads=2)
    np.random.seed(24680)
    rng_before = np.random.get_state()

    convert_gpt_kv_heads(source, 1)

    _rng_state_equal(np.random.get_state(), rng_before)


def test_extreme_same_sign_head_mean_does_not_overflow_under_strict_numpy_errors():
    source = _model()
    maximum = np.finfo(np.float64).max
    rows = source.blocks[0].attn.W_k.weight.data.reshape(4, 2, 8)
    rows[0].fill(maximum)
    rows[1].fill(maximum)
    rows[2].fill(-maximum)
    rows[3].fill(-maximum)

    with np.errstate(all="raise"):
        converted = convert_gpt_kv_heads(source, 2)

    result = converted.blocks[0].attn.W_k.weight.data.reshape(2, 2, 8)
    np.testing.assert_array_equal(result[0], np.full((2, 8), maximum))
    np.testing.assert_array_equal(result[1], np.full((2, 8), -maximum))


def test_extreme_opposite_sign_head_mean_cancels_without_overflow():
    source = _model()
    maximum = np.finfo(np.float64).max
    rows = source.blocks[0].attn.W_v.weight.data.reshape(4, 2, 8)
    rows[0].fill(maximum)
    rows[1].fill(-maximum)
    rows[2].fill(maximum)
    rows[3].fill(-maximum)

    with np.errstate(all="raise"):
        converted = convert_gpt_kv_heads(source, 2)

    result = converted.blocks[0].attn.W_v.weight.data
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_smallest_subnormal_rows_convert_without_underflow_warning():
    source = _model()
    tiny = np.nextafter(0.0, 1.0)
    rows = source.blocks[0].attn.W_k.weight.data.reshape(4, 2, 8)
    rows[0].fill(tiny)
    rows[1].fill(tiny)
    rows[2].fill(-tiny)
    rows[3].fill(-tiny)

    with np.errstate(all="raise"):
        converted = convert_gpt_kv_heads(source, 2)

    result = converted.blocks[0].attn.W_k.weight.data.reshape(2, 2, 8)
    np.testing.assert_array_equal(result[0], np.full((2, 8), tiny))
    np.testing.assert_array_equal(result[1], np.full((2, 8), -tiny))
