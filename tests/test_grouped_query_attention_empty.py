import numpy as np
import pytest

from engine.tensor import Tensor
from nn import GroupedQueryAttention


@pytest.mark.parametrize("shape", [(0, 2, 8), (1, 0, 8)])
def test_forward_rejects_empty_batch_or_time(shape):
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2)
    with pytest.raises(ValueError, match="batch and time must be non-empty"):
        attention(Tensor(np.zeros(shape)))


@pytest.mark.parametrize("shape", [(0, 2, 8), (1, 0, 8)])
def test_infer_rejects_empty_batch_or_time(shape):
    attention = GroupedQueryAttention(8, 4, num_kv_heads=2)
    with pytest.raises(ValueError, match="batch and time must be non-empty"):
        attention.infer(np.zeros(shape))
