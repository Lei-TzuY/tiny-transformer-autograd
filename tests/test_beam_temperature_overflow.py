import numpy as np
import pytest

from nn import beam_generate


class _NoInferenceModel:
    context_len = 4

    def __init__(self):
        self.validation_calls = 0
        self.infer_calls = 0

    def _validate_token_batch(self, idx):
        self.validation_calls += 1
        values = np.asarray(idx)
        if values.ndim != 2:
            raise ValueError("token ids must have shape (batch, time)")
        return values

    def _validate_generation_mask(self, attention_mask, idx_shape):
        raise AssertionError("overflow validation must happen before mask validation")

    def infer(self, *args, **kwargs):
        self.infer_calls += 1
        raise AssertionError("overflow validation must happen before inference")


@pytest.mark.parametrize("temperature", [10**400, -(10**400)])
def test_direct_beam_temperature_overflow_is_normalized_before_model_use(temperature):
    model = _NoInferenceModel()

    with pytest.raises(ValueError, match=r"^temperature must be finite$"):
        beam_generate(model, [[1, 2]], 1, temperature=temperature)

    assert model.validation_calls == 0
    assert model.infer_calls == 0


def test_large_representable_temperature_remains_supported_for_zero_token_request():
    model = _NoInferenceModel()
    prompt = np.array([[1, 2]], dtype=np.int64)

    result = beam_generate(model, prompt, 0, temperature=10**300)

    np.testing.assert_array_equal(result, prompt)
    assert result is not prompt
    assert model.validation_calls == 1
    assert model.infer_calls == 0


def test_existing_non_finite_temperature_error_is_unchanged():
    model = _NoInferenceModel()

    with pytest.raises(ValueError, match=r"^temperature must be finite$"):
        beam_generate(model, [[1, 2]], 0, temperature=np.inf)

    assert model.validation_calls == 0
    assert model.infer_calls == 0
