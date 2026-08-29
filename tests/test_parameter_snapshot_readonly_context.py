import numpy as np

from engine.parameter_snapshot import ParameterSnapshot
from engine.tensor import Tensor


def test_installed_exit_recovers_when_body_makes_live_storage_read_only():
    p = Tensor([3.0, 4.0], requires_grad=True)
    snapshot = ParameterSnapshot(p, values=np.array([1.0, 2.0]))

    with snapshot.installed():
        np.testing.assert_array_equal(p.data, [1.0, 2.0])
        p.data.flags.writeable = False

    assert p.data.flags.writeable is True
    np.testing.assert_array_equal(p.data, [3.0, 4.0])
