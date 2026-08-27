"""Train with the non-executable safe checkpoint format.

This is a deliberately thin adapter around :mod:`train`. The training loop owns
all model, optimizer, scheduler, RNG, data, and generation semantics; this module
only swaps its checkpoint reader/writer for the NPZ/JSON safe format during one
CLI invocation. Restoring the original callables in ``finally`` keeps in-process
tests and embedders isolated after success, ``--help``, or an exception.
"""

import train
from engine.safe_checkpoint import read_safe_checkpoint, save_safe_checkpoint


def main():
    """Run ``tiny-train`` semantics using safe checkpoint persistence only."""
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
