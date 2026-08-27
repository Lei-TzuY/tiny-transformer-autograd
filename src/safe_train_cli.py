"""Train with the non-executable safe checkpoint format.

This is a deliberately thin adapter around :mod:`train`. The training loop owns
all model, optimizer, scheduler, RNG, data, and generation semantics; this module
only swaps its checkpoint reader/writer for the NPZ/JSON safe format during one
CLI invocation. A process-local reentrant lock keeps overlapping in-process safe
CLI calls from interleaving their temporary global swap. Restoring the original
callables in ``finally`` keeps tests and embedders isolated after success,
``--help``, or an exception.
"""

import threading

import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


_checkpoint_io_lock = threading.RLock()


def main():
    """Run ``tiny-train`` semantics using safe checkpoint persistence only."""
    # ``train.main`` resolves checkpoint I/O through module globals. Serialize
    # the whole temporary swap so two embedded safe invocations cannot restore
    # those globals out of order. RLock keeps same-thread nested calls usable.
    with _checkpoint_io_lock:
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
