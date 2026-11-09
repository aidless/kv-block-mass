"""Tests for the byte ledger.

The ledger exists because "same budget" comparisons can compare different
numbers of bytes: a per-head selection stored as a mask over a dense tensor
never frees anything. These tests pin the arithmetic, then pin the verdict words
so a results table cannot silently start claiming enforcement it does not have.
"""

import math

import pytest

from kvblockmass.budget import (
    CacheLayout,
    audit_layout,
    bytes_full,
    bytes_mask_stored,
    bytes_ragged,
)


def _layout(**kw):
    base = dict(n_layers=28, n_heads=8, seq_len=4096, head_dim=128, item_bytes=2, block_size=16)
    base.update(kw)
    return CacheLayout(**base)


def test_full_cache_arithmetic():
    lay = _layout()
    # 2 (K and V) * 28 * 8 * 4096 * 128 * 2 bytes
    assert bytes_full(lay) == 2 * 28 * 8 * 4096 * 128 * 2
    assert lay.per_token_bytes == 2 * 128 * 2


def test_mask_layout_holds_the_full_cache():
    """The failure mode the source paper found, reproduced in arithmetic."""
    lay = _layout()
    mask_bytes = math.ceil(lay.seq_len / 8) * lay.n_layers * lay.n_heads
    assert bytes_mask_stored(lay) == bytes_full(lay) + mask_bytes
    audit = audit_layout(lay, kept_per_head=1024, mask_stored=True)
    assert audit.holds_full_cache is True
    assert audit.fraction_as_stored > 0.99        # measured, not asserted by name
    assert audit.as_row()["holds_full_cache"] is True


def test_ragged_layout_enforces_the_nominal_budget():
    lay = _layout()
    audit = audit_layout(lay, kept_per_head=1024, mask_stored=False)   # 25% of 4096
    assert audit.holds_full_cache is False
    assert audit.nominal_is_enforced is True
    # 25% payload + block-table overhead, so slightly above a quarter
    assert 0.25 <= audit.fraction_enforced < 0.26


def test_block_table_overhead_is_small_but_not_zero():
    lay = _layout()
    kept = 1024
    payload = lay.n_layers * lay.n_heads * kept * lay.per_token_bytes
    blocks = math.ceil(kept / lay.block_size)
    expected = payload + lay.n_layers * lay.n_heads * blocks * 4
    assert bytes_ragged(lay, kept) == expected
    assert bytes_ragged(lay, kept) > payload


def test_dense_storage_states_a_budget_it_does_not_hold():
    """A policy can claim 25% and store 100% — the ledger must show both numbers.

    ``nominal_is_enforced`` describes the *ragged* layout only; the misstatement
    lives in ``fraction_as_stored``. Both are exposed so a results table cannot
    quote the flattering one.
    """
    lay = _layout()
    audit = audit_layout(lay, kept_per_head=1024, mask_stored=True)
    assert 0.24 < audit.fraction_enforced < 0.26      # the choice is at budget
    assert audit.fraction_as_stored > 0.99            # the bytes are not
    assert audit.fraction_as_stored > 3.0 * audit.nominal_fraction
    assert audit.holds_full_cache is True


def test_ragged_arithmetic_is_linear_in_kept_tokens():
    lay = _layout()
    # equal token deltas *and* equal block-table deltas (16 blocks each), so the
    # payload term is isolated from the step-wise block-table term
    a = bytes_ragged(lay, 512) - bytes_ragged(lay, 256)
    b = bytes_ragged(lay, 768) - bytes_ragged(lay, 512)
    assert a == b
    assert a == lay.n_layers * lay.n_heads * 256 * lay.per_token_bytes + 16 * lay.n_layers * lay.n_heads * 4


def test_edge_cases_zero_and_full_retention():
    lay = _layout()
    assert bytes_ragged(lay, 0) == 0
    assert bytes_ragged(lay, lay.seq_len) >= bytes_full(lay)   # plus tables
    full = audit_layout(lay, kept_per_head=lay.seq_len, mask_stored=False)
    assert full.nominal_is_enforced is True


def test_invalid_layouts_raise():
    with pytest.raises(ValueError):
        _layout(n_layers=0)
    with pytest.raises(ValueError):
        _layout(item_bytes=0)
    with pytest.raises(ValueError):
        bytes_ragged(_layout(), -1)


def test_committed_ledger_json_matches_library():
    """The README ledger table is checked against this file. Recompute it."""
    import json
    from pathlib import Path

    from tests.cpu_observe import ledger_snapshot

    path = Path(__file__).resolve().parents[1] / "bench" / "cpu_ledger.json"
    data = json.loads(path.read_text())
    live = ledger_snapshot()
    assert live["layout"] == data["layout"]
    assert live["kept_per_head"] == data["kept_per_head"]
    assert live["bytes_full"] == data["bytes_full"]
    assert live["mask_overhead_bytes"] == data["mask_overhead_bytes"]
    assert live["mask"] == data["mask"]
    assert live["ragged"] == data["ragged"]
    assert data["full_mib"] == live["bytes_full"] / (1024 ** 2)


def test_audit_row_is_json_ready():
    import json
    row = audit_layout(_layout(), 1024, mask_stored=False).as_row()
    json.dumps(row)                    # must not raise (numpy types would)
    assert set(row) == {
        "kept_per_head", "nominal_frac", "as_stored_frac",
        "enforced_frac", "holds_full_cache", "nominal_is_enforced",
    }
