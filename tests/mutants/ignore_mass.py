"""Mutant IGNORE_MASS: the plan stops reading the kernel's block mass.

Broadcasting a constant instead of the measured mass makes the design uniform:
everything still works, the card still has numbers, and the policy silently stops
caring which blocks matter. Must be caught by the empty-half test.
"""

import torch

from kvblockmass import reclaim as _r


def _patched(block_scores, seq_len, block_size):
    return torch.ones(seq_len, dtype=torch.float64)


_r.expand_block_scores = _patched
