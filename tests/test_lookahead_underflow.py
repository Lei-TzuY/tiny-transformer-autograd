import numpy as np

from engine.lookahead import Lookahead
from engine.optim import SGD
from engine.tensor import Tensor


def test_subnormal_interpolation_is_warning_neutral_under_strict_errstate():
    tiny = np.nextafter(0.0, 1.0)
    parameter = Tensor([tiny], requires_grad=True)
    parameter.grad = None
    optimizer = Lookahead(SGD([parameter]), sync_period=7, alpha=0.5)
    parameter.data[...] = [0.0]

    with np.errstate(all="raise"):
        optimizer.sync()

    assert np.isfinite(parameter.data).all()
    assert np.isfinite(optimizer.slow_weights()[0]).all()
