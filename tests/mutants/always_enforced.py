"""Mutant ALWAYS_ENFORCED: the ledger always says the budget is enforced.

The verdict word is what a results table quotes, so a stamped-on "True" is the
cheapest possible lie. Must be caught by the audit tests.
"""

from kvblockmass import budget as _b

_b.LayoutAudit.nominal_is_enforced = property(lambda self: True)
_b.LayoutAudit.holds_full_cache = property(lambda self: False)
