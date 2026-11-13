"""End-to-end policy card: synthetic Q/K -> block mass -> plan -> bytes.

Run:  PYTHONPATH=src python examples/policy_card.py

Everything here runs on CPU in about a second. The point is to show the whole
chain in one place and to print the numbers a results table would quote — kept
tokens, bytes, and what the certificate costs — rather than a "budget %" that
can mean three different things.
"""

import json

import torch

from kvblockmass.budget import CacheLayout
from kvblockmass.certificate import error_certificate, full_cache_output, relative_error
from kvblockmass.reclaim import plan_reclaim
from kvblockmass.reference import block_mass_reference

N, BLOCK, D = 512, 16, 64
torch.manual_seed(0)

# 1. block mass as the kernel produces it (CPU reference here; triton on GPU)
q = torch.randn(1, 8, N, D, dtype=torch.float64)
k = torch.randn(1, 2, N, D, dtype=torch.float64)
mass = block_mass_reference(q, k, causal=True, block_size=BLOCK)   # [num_blocks]

# 2. a plan at a stated budget
layout = CacheLayout(n_layers=4, n_heads=8, seq_len=N, head_dim=D, item_bytes=2, block_size=BLOCK)
plan = plan_reclaim(
    mass, BLOCK, layout, tail_budget=N // 8,
    n_sinks=4, recency=32, top_slice=N // 16, seed=0,
)
print("policy card")
print(json.dumps(plan.card, indent=2, sort_keys=True))

# 3. what the certificate says about one head of that plan
# one head's logits: the same weights the design was built from, so the
# certificate is judged on the cache it actually describes (not on fresh noise)
from kvblockmass.evict import expand_block_scores
scores = torch.log(expand_block_scores(mass, N, BLOCK).clamp_min(1e-30))
values = torch.randn(N, D, dtype=torch.float64)
cert = error_certificate(scores, values, plan.design.keep, plan.design.pi, plan.design.certainty)
realized = relative_error(cert.y_hat, full_cache_output(scores, values))
print(f"\ncertificate radius {cert.radius:.4f}  realized error {realized:.4f}  "
      f"{'covered' if realized <= cert.radius else 'NOT COVERED'}")

print("\nreading:")
print(f"  kept {plan.kept_tokens}/{N} tokens = {100 * plan.card['kept_fraction']:.1f}% of tokens,")
print(f"  but as stored the policy holds {100 * plan.card['as_stored_frac']:.2f}% of the cache")
print(f"  (enforced {100 * plan.card['enforced_frac']:.2f}% only if the selection is ragged)")
