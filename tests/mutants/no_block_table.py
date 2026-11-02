"""Mutant NO_BLOCK_TABLE: ragged storage forgets its block-table overhead.

Small enough to look free, large enough to make a "same budget" comparison wrong.
Must be caught by the ledger arithmetic.
"""

import math

from kvblockmass import budget as _b

_orig = _b.bytes_ragged


def _patched(layout, kept_per_head):
    if kept_per_head < 0:
        raise ValueError("kept_per_head must be >= 0")
    kept = min(kept_per_head, layout.seq_len)
    return int(layout.n_layers * layout.n_heads * kept * layout.per_token_bytes)


_b.bytes_ragged = _patched
