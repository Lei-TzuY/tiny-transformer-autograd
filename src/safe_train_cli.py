"""Train with the non-executable safe checkpoint format.

This is a deliberately thin adapter around :mod:`train`. The training loop owns
all model, optimizer, scheduler, RNG, data, and generation semantics; this module
only swaps its checkpoint reader/writer for the NPZ/JSON safe format during one
CLI invocation. Restoring the original callables in ``finally`` keeps in-process
tests and embedders isolated after success, ``--help``, or an exception.
"""

import threading

import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


_CHECKPOINT_IO_PATCH_LOCK = threading.RLock()


def main():
    """Run ``tiny-train`` semantics using safe checkpoint persistence only."""
    # The adapter temporarily mutates process-global functions on ``train``.
    # Serialize complete invocations so one in-process caller cannot restore
    # pickle I/O while another safe-training call is still using the module.
    with _CHECKPOINT_IO_PATCH_LOCK:
        original_reader = train.read_checkpoint
        original_writer = train.save_checkpoint
        train.read_checkpoint = read_safe_checkpoint
        train.save_checkpoint = save_safe_checkpoint
        try:
            return train.main()
        finally:
            train.read_checkpoint = original_reader
            train.save_checkpoint = original_writer


if __name__ == "__main__":
    main()
