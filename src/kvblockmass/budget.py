"""Byte accounting: what an eviction policy *holds* vs the budget it *claims*.

arXiv:2608.25230 makes an uncomfortable observation: in a shared evaluation
pipeline, the strongest query-agnostic eviction methods held the **full** KV
cache, because their per-head selections were stored as masks over a
dense-allocated tensor — nothing was ever freed — while the methods that looked
worse had actually released memory. Enforcing a nominal budget on one fixed
selection then cost 14 to 62 benchmark points. In other words: a comparison at
"the same budget" can compare different numbers of bytes.

This module makes the accounting explicit and runnable, so a policy card can
state bytes rather than a ratio:

    full      = 2 * L * H * N * D * item_size            (K and V, dense)
    mask      = full + ceil(N / 8) * L * H               (selection as a mask)
    ragged    = 2 * L * H * kept_per_head_avg * D * item  (+ block tables)

Ragged storage is the only variant where the budget is physically enforced.
The audit returns both numbers and a verdict, so a results table can be checked
against it instead of trusted.

Not in scope: the *quality* cost of enforcing a budget (their 14–62 benchmark
points). That needs a model and a benchmark, and we do not run one here — see
``docs/evolution-from-papers.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CacheLayout",
    "bytes_full",
    "bytes_mask_stored",
    "bytes_ragged",
    "audit_layout",
]

# Block-table overhead per (layer, head, block): one 32-bit block id.
_BLOCK_ID_BYTES = 4


@dataclass(frozen=True)
class CacheLayout:
    """Shape of the KV cache being accounted for."""

    n_layers: int
    n_heads: int
    seq_len: int
    head_dim: int
    item_bytes: int = 2       # fp16 by default
    block_size: int = 16

    def __post_init__(self) -> None:
        for name in ("n_layers", "n_heads", "seq_len", "head_dim", "block_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.item_bytes <= 0:
            raise ValueError("item_bytes must be positive")

    @property
    def per_token_bytes(self) -> int:
        """K plus V for one token in one head."""
        return 2 * self.head_dim * self.item_bytes


def bytes_full(layout: CacheLayout) -> int:
    """Dense KV cache for every token in every head."""
    return (
        layout.n_layers
        * layout.n_heads
        * layout.seq_len
        * layout.per_token_bytes
    )


def bytes_mask_stored(layout: CacheLayout) -> int:
    """Dense cache *plus* a per-head selection mask — memory is never freed."""
    mask = math.ceil(layout.seq_len / 8) * layout.n_layers * layout.n_heads
    return bytes_full(layout) + mask


def bytes_ragged(layout: CacheLayout, kept_per_head: float) -> int:
    """Physically evicted storage: only the kept tokens, plus block tables."""
    if kept_per_head < 0:
        raise ValueError("kept_per_head must be >= 0")
    kept = min(kept_per_head, layout.seq_len)
    payload = (
        layout.n_layers * layout.n_heads * kept * layout.per_token_bytes
    )
    blocks = math.ceil(kept / layout.block_size) if kept else 0
    tables = layout.n_layers * layout.n_heads * blocks * _BLOCK_ID_BYTES
    return int(payload + tables)


@dataclass
class LayoutAudit:
    layout: CacheLayout
    kept_per_head: float
    nominal_fraction: float        # the budget the policy claims
    bytes_full: int
    bytes_as_stored: int
    bytes_enforced: int

    @property
    def fraction_as_stored(self) -> float:
        return self.bytes_as_stored / self.bytes_full

    @property
    def fraction_enforced(self) -> float:
        return self.bytes_enforced / self.bytes_full

    @property
    def holds_full_cache(self) -> bool:
        """True when the stored layout never frees memory (mask over dense)."""
        return self.fraction_as_stored > 0.95

    @property
    def nominal_is_enforced(self) -> bool:
        """True when the enforced layout is within 5% of the claimed budget."""
        target = self.nominal_fraction * self.bytes_full
        return abs(self.bytes_enforced - target) <= 0.05 * self.bytes_full

    def as_row(self) -> dict:
        return {
            "kept_per_head": round(self.kept_per_head, 2),
            "nominal_frac": round(self.nominal_fraction, 4),
            "as_stored_frac": round(self.fraction_as_stored, 4),
            "enforced_frac": round(self.fraction_enforced, 4),
            "holds_full_cache": self.holds_full_cache,
            "nominal_is_enforced": self.nominal_is_enforced,
        }


def audit_layout(
    layout: CacheLayout,
    kept_per_head: float,
    *,
    mask_stored: bool = True,
) -> LayoutAudit:
    """Account one policy's memory under both storage strategies.

    ``mask_stored=True`` models the failure mode the source paper found: the
    selection is a mask over a dense tensor, so the claimed budget is not
    physically enforced even though the *choice* is at that budget.
    """
    return LayoutAudit(
        layout=layout,
        kept_per_head=kept_per_head,
        nominal_fraction=kept_per_head / layout.seq_len,
        bytes_full=bytes_full(layout),
        bytes_as_stored=bytes_mask_stored(layout) if mask_stored else bytes_ragged(layout, kept_per_head),
        bytes_enforced=bytes_ragged(layout, kept_per_head),
    )
