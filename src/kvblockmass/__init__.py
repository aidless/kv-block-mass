"""kvblockmass: block-granular attention mass in O(N) memory.

Companion of https://github.com/aidless/paged-kv — paged-kv manages block
lifetimes (refcount / COW / failure-side-effect-free); this package computes
*which* blocks deserve to live, i.e. the block-level attention mass used by
KV-eviction research (H2O / SnapKV style scoring), without materializing the
[B, H, N, N] attention matrix.
"""

__version__ = "0.1.0"

from .reference import block_mass_reference

__all__ = ["block_mass_reference", "__version__"]
