"""Cross-process determinism regression for complete Transformer gradients."""

import os
import subprocess
import sys
import textwrap


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")

_SCRIPT = textwrap.dedent(
    r"""
    import hashlib
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.environ["TINY_TRANSFORMER_SRC"])

    import engine.ops as ops
    from nn.transformer import GPT

    # Keep these allocations alive so Tensor object addresses differ across
    # subprocesses. Ordered parent traversal must make that irrelevant.
    perturbation = int(os.environ["GRAPH_ADDRESS_PERTURBATION"])
    _address_junk = [object() for _ in range(perturbation)]


    def _update_array(digest, array):
        array = np.asarray(array)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))


    def gradient_digest(label, architecture):
        np.random.seed(20260825)
        model = GPT(
            vocab_size=7,
            context_len=3,
            d_model=4,
            num_heads=2,
            d_ff=8,
            num_layers=2,
            dropout=0.0,
            **architecture,
        )
        tokens = np.array([[0, 1, 2], [2, 3, 1]], dtype=np.int64)
        targets = np.array([[1, 2, 3], [3, 1, 0]], dtype=np.int64)
        loss = ops.cross_entropy(model(tokens), targets)
        loss.backward()

        digest = hashlib.sha256()
        digest.update(label.encode("utf-8"))
        _update_array(digest, np.asarray(loss.data, dtype=np.float64))
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            digest.update(name.encode("utf-8"))
            _update_array(digest, parameter.data)
            _update_array(digest, parameter.grad)
        return digest.hexdigest()


    results = [
        gradient_digest(
            "gpt",
            {"norm": "layernorm", "pos_encoding": "learned", "ffn": "gelu"},
        ),
        gradient_digest(
            "llama",
            {"norm": "rmsnorm", "pos_encoding": "rope", "ffn": "swiglu"},
        ),
    ]
    print(" ".join(results))
    """
)


def _run_digest(perturbation):
    env = os.environ.copy()
    env["GRAPH_ADDRESS_PERTURBATION"] = str(perturbation)
    env["PYTHONHASHSEED"] = str(perturbation + 1)
    env["TINY_TRANSFORMER_SRC"] = _SRC
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    return subprocess.check_output(
        [sys.executable, "-W", "error", "-c", _SCRIPT],
        cwd=_ROOT,
        env=env,
        text=True,
    ).strip()


def test_full_transformer_gradients_are_cross_process_deterministic():
    digests = [_run_digest(count) for count in (0, 37, 211)]

    assert digests[0] == digests[1] == digests[2]
    gpt_digest, llama_digest = digests[0].split()
    assert len(gpt_digest) == len(llama_digest) == 64
    assert gpt_digest != llama_digest
