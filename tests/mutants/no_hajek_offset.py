"""Mutant NO_HAJEK_OFFSET: the design is collected but the estimator ignores it.

This is the classic "we sampled proportionally, then forgot to reweight" bug: the
retained set is drawn from the design, but ``pi`` never enters the estimate, so
the result is a plain (biased) subsample mean. Must be caught by the unbiasedness
test.
"""

import torch

from kvblockmass import certificate as _c

_orig = _c.hajek_output


def _patched(scores, values, keep, pi):
    return _orig(scores, values, keep, torch.ones_like(pi))


_c.hajek_output = _patched
