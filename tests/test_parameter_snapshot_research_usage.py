import numpy as np

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_snapshot_supports_symmetric_finite_difference_style_evaluation():
    p = Tensor(2.0, requires_grad=True)
    plus = ParameterSnapshot(p, values=np.array(2.5))
    minus = ParameterSnapshot(p, values=np.array(1.5))

    def objective():
        return float(np.asarray(p.data) ** 2)

    baseline = objective()
    with plus.installed():
        plus_value = objective()
    with minus.installed():
        minus_value = objective()

    assert baseline == 4.0
    assert plus_value == 6.25
    assert minus_value == 2.25
    assert p.data.item() == 2.0
    assert (plus_value - minus_value) / 1.0 == 4.0
