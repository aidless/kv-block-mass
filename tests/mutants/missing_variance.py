"""Mutant MISSING_VARIANCE: the radius keeps only the range term.

The certificate then reports a radius that ignores sampling variance, so coverage
collapses below 1-delta. Must be caught by the coverage test.
"""

import math

from kvblockmass import certificate as _c

_orig = _c.error_certificate


def _patched(*args, delta=0.1, **kwargs):
    cert = _orig(*args, delta=delta, **kwargs)
    cert.variance = 0.0
    log1d = math.log(1.0 / delta)
    cert.radius = cert.range_term * log1d / (float(cert.y_hat.norm()) + 1e-12)
    return cert


_c.error_certificate = _patched
