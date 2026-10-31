"""Tests for M6 (valuation)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import value as V


def test_probabilities_sum_to_one_and_follow_the_gap():
    rng = np.random.default_rng(1)
    gain = rng.normal(0.0, 1.0, 5000)
    cheap = V.verdict_probs(gain, 1e6, 0.07)       # claims 0.07 points: fair
    dear = V.verdict_probs(gain, 100e6, 0.07)      # claims 7 points: overpaid
    for r in (cheap, dear):
        assert abs(r["p_overpaid"] + r["p_fair"] + r["p_best_buy"] - 1) < 1e-9
    assert cheap["verdict"] == "Fair" and dear["verdict"] == "Overpaid"
    star = V.verdict_probs(rng.normal(6, 1, 5000), 10e6, 0.07)
    assert star["verdict"] == "Best Buy" and star["confidence"] > 0.95


def test_more_uncertainty_means_less_confidence():
    rng = np.random.default_rng(2)
    sharp = V.verdict_probs(rng.normal(-3, 0.5, 5000), 20e6, 0.07)
    vague = V.verdict_probs(rng.normal(-3, 6, 5000), 20e6, 0.07)
    assert sharp["confidence"] > vague["confidence"]


def test_higher_fee_never_improves_the_verdict():
    gain = np.random.default_rng(3).normal(1.0, 2.0, 5000)
    p = [V.verdict_probs(gain, f * 1e6, 0.07)["p_best_buy"] for f in (5, 20, 50, 100)]
    assert all(a >= b for a, b in zip(p, p[1:]))


def test_within_club_slope_removes_rich_club_effect():
    """Rich clubs are better (between-club slope is big) but within a club, value changes do not move points."""
    import pandas as pd
    rng = np.random.default_rng(7)
    rows = []
    for c in range(30):
        base_value, base_pts = 50 + 30 * c, 30 + 1.5 * c          # richer club -> better club
        for s in range(2015, 2023):
            rows.append({"club_id": c, "league": "GB1", "season": s,
                         "value_m": base_value + rng.normal(0, 20), "points": base_pts + rng.normal(0, 3)})
    x = pd.DataFrame(rows)
    between = V.slope_from_panel(x, within_club=False, n_boot=20)["b"]
    within = V.slope_from_panel(x, within_club=True, n_boot=20)["b"]
    assert between > 0.04 and abs(within) < 0.02


def test_quantile_verdict_matches_draw_verdict():
    rng = np.random.default_rng(11)
    gain = rng.normal(1.0, 2.0, 20000)
    q = np.percentile(gain, V.QUANTILE_LEVELS)
    for fee, b in [(10e6, 0.014), (60e6, 0.014), (60e6, 0.069), (120e6, 0.069)]:
        a, d = V.verdict_from_quantiles(q, fee, b), V.verdict_probs(gain, fee, b)
        assert abs(a["p_overpaid"] - d["p_overpaid"]) < 0.03 and abs(a["p_best_buy"] - d["p_best_buy"]) < 0.03
        assert abs(a["p_overpaid"] + a["p_fair"] + a["p_best_buy"] - 1) < 1e-9
