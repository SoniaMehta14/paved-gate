"""Full Bitext benchmark helpers: data splits, checkpoint records, cost accounting and the paired test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "support_tickets"))
sys.path.insert(0, str(ROOT / "scripts"))

import full_bench  # noqa: E402
from build_sample import DEFAULT_CACHE  # noqa: E402
from pricing import cost_usd  # noqa: E402

needs_dataset = pytest.mark.skipif(not DEFAULT_CACHE.exists(), reason="Bitext CSV not downloaded")


@needs_dataset
def test_splits_are_deterministic_and_exemplars_never_evaluated() -> None:
    a, b = full_bench.build_data(50), full_bench.build_data(50)
    assert [t.row_index for t in a.exemplars] == [t.row_index for t in b.exemplars]
    assert [t.row_index for t in a.subset] == [t.row_index for t in b.subset]
    ex = {t.row_index for t in a.exemplars}
    assert len(ex) == 27 and len({t.intent for t in a.exemplars}) == 27
    assert not ex & {t.row_index for t in a.eval_set}
    assert len(a.eval_set) == len(a.rows) - 27 == len({t.row_index for t in a.eval_set})
    assert {t.row_index for t in a.flagship} <= {t.row_index for t in a.subset}
    assert [t.row_index for t in a.eval_set[: len(a.subset)]] == [t.row_index for t in a.subset]  # subset first
    assert {t.row_index for t in a.smoke} <= {t.row_index for t in a.subset} and len(a.smoke) == 50


def test_latest_record_per_row_wins() -> None:
    recs: list[dict[str, object]] = [
        {"row": 1, "pred": None, "error": "timeout"},
        {"row": 2, "pred": "x"},
        {"row": 1, "pred": "y"},
    ]
    latest = full_bench.latest_by_row(recs)
    assert latest[1]["pred"] == "y" and set(latest) == {1, 2}


def test_record_cost_uses_cached_rate() -> None:
    rec: dict[str, object] = {"input_tokens": 1_000_000, "cached_tokens": 600_000, "output_tokens": 0}
    assert full_bench.record_cost(rec, "gpt-4o") == pytest.approx(0.4 * 2.50 + 0.6 * 1.25)
    assert cost_usd("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)  # existing callers unchanged


def test_mcnemar_exact() -> None:
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2
    t = full_bench.mcnemar(a, b)
    assert (t["a_only"], t["b_only"]) == (10, 2) and t["p_value"] == pytest.approx(0.038574, abs=1e-5)
    assert full_bench.mcnemar([True], [True])["p_value"] == 1.0
