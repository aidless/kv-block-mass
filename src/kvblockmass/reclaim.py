"""The glue layer: block mass in, a reclaim plan with stated bytes out.

Everything before this file is a layer: the kernel measures, ``evict`` designs,
``certificate`` bounds, ``budget`` accounts. A *policy* is the composition, and
composition is where libraries usually go quiet — each piece is correct and the
joined-up thing is not.

``plan_reclaim`` takes what the kernel actually produces (per-**block** mass) and
returns a token-level plan whose byte cost is stated, so a results table can
quote ``kept_tokens`` and ``bytes_enforced`` from the same object that produced
the estimate. Block mass is broadcast to tokens inside ``expand_block_scores``;
ties inside a block are broken by the sampling design, not by position.

Not in scope: the scheduler that would actually free the blocks (that is
``aidless/paged-kv``), and the quality cost of the budget (needs a model and a
benchmark — see ``docs/evolution-from-papers.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .budget import CacheLayout, LayoutAudit, audit_layout
from .evict import EvictionDesign, design_from_scores, expand_block_scores

__all__ = ["ReclaimPlan", "plan_reclaim"]


@dataclass
class ReclaimPlan:
    """A realized plan: the design, the realized tokens, the bytes it holds."""

    design: EvictionDesign
    audit: LayoutAudit
    kept_tokens: int
    card: dict

    @property
    def blocks_kept(self) -> int:
        return int(self.card["blocks_kept"])


def _block_retention(keep: torch.Tensor, seq_len: int, block_size: int) -> tuple[int, int]:
    """(blocks_kept, blocks_total) — a block counts as kept if any token is kept."""
    num_blocks = math.ceil(seq_len / block_size)
    blocks_kept = 0
    for b in range(num_blocks):
        start = b * block_size
        stop = min(start + block_size, seq_len)
        if bool(keep[start:stop].any()):
            blocks_kept += 1
    return blocks_kept, num_blocks


def plan_reclaim(
    block_mass: torch.Tensor,          # [num_blocks] — output of the kernel
    block_size: int,
    layout: CacheLayout,
    tail_budget: int,
    *,
    n_sinks: int = 4,
    recency: int = 32,
    top_slice: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    mask_stored: bool = True,
) -> ReclaimPlan:
    """Turn block mass into a token-level design with a stated byte cost.

    ``tail_budget`` is the *expected* number of randomly retained tail tokens;
    the certainty set (score top slice + sinks + recency) is deterministic on top
    of it, so the realized token count is ``certainty + realized tail``.
    ``mask_stored=True`` models the common failure mode where the selection is a
    mask over a dense tensor; set it to ``False`` to model ragged storage.
    """
    if block_mass.dim() != 1:
        raise ValueError(f"block_mass must be 1-D, got {tuple(block_mass.shape)}")
    if bool((block_mass < 0).any()):
        raise ValueError("block_mass must be non-negative")
    seq_len = layout.seq_len
    if block_mass.numel() * block_size < seq_len:
        raise ValueError(
            f"{block_mass.numel()} blocks of {block_size} cannot cover seq_len={seq_len}"
        )
    if tail_budget < 0:
        raise ValueError("tail_budget must be >= 0")

    scores = expand_block_scores(block_mass, seq_len, block_size)
    top = tail_budget // 2 if top_slice is None else max(0, top_slice)
    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(seed)

    design = design_from_scores(
        scores,
        tail_budget,
        n_sinks=n_sinks,
        recency=recency,
        top_slice=top,
        generator=generator,
    )
    kept_tokens = int(design.keep.sum())
    audit = audit_layout(layout, kept_tokens, mask_stored=mask_stored)
    blocks_kept, blocks_total = _block_retention(design.keep, seq_len, block_size)

    card = {
        "kept_tokens": kept_tokens,
        "kept_fraction": round(kept_tokens / seq_len, 4),
        # absolute bytes, not just ratios: a "budget" column hides the difference
        "bytes_full": audit.bytes_full,
        "bytes_as_stored": audit.bytes_as_stored,
        "bytes_enforced": audit.bytes_enforced,
        "tail_budget": int(tail_budget),
        "top_slice": int(top),
        "n_sinks": int(n_sinks),
        "recency": int(recency),
        "blocks_total": blocks_total,
        "blocks_kept": blocks_kept,
        "block_fraction": round(blocks_kept / blocks_total, 4) if blocks_total else 0.0,
        **audit.as_row(),
    }
    return ReclaimPlan(design=design, audit=audit, kept_tokens=kept_tokens, card=card)
