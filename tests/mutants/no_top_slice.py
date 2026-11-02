"""Mutant NO_TOP_SLICE: the certainty set loses its score top slice.

Budget is still spent, but the heaviest tokens are now sampled instead of kept,
so the variance explodes and the design loses to top-k everywhere. Must be caught
by the regime and unbiasedness tests. (This was a real bug in the first draft of
the design: the certainty set was purely positional.)
"""

from kvblockmass import evict as _e

_orig = _e.design_from_scores


def _patched(scores, tail_budget, **kwargs):
    kwargs["top_slice"] = 0
    return _orig(scores, tail_budget, **kwargs)


_e.design_from_scores = _patched
