"""Tests for the 30 Oct 2026 changes: weakest-first displacement (M5 v2), the promoted-club rule, and the
2026/27 forecast script."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeagent import simulate as M


def squad():
    # club 1: three midfielders (ratings 0.30, 0.10, -0.05) and a defender; club 2 elsewhere
    return pd.DataFrame({"club_id": [1, 1, 1, 1, 1, 2],
                         "player_id": [10, 11, 12, 13, 99, 20],
                         "share": [0.9, 0.6, 0.3, 1.0, 0.4, 1.0],
                         "rating": [0.30, 0.10, -0.05, 0.20, 0.15, 0.0],
                         "position": ["Midfield", "Midfield", "Midfield", "Defender", "Midfield", "Attack"]})


def test_without_player_removes_him_and_keeps_the_club_total():
    sq = squad()
    rest = M.without_player(sq, 1, 99)
    assert 99 not in set(rest.player_id)
    assert np.isclose(rest.share.sum(), sq[sq.club_id == 1].share.sum())
    assert set(rest.club_id) == {1}


def test_signing_takes_minutes_from_the_weakest_player_first():
    rest = M.without_player(squad(), 1, 99)
    order = M.displacement_order(rest, "Midfield")
    small = rest[rest.player_id == 12].share.iloc[0] / 2       # fewer minutes than the weakest midfielder has
    lost = M.displaced_mass(order, np.array([small]))[0]
    assert np.isclose(lost / small, -0.05)                     # he replaces the -0.05 player, not the average
    avg = M.displaced_rating(rest, 1, "Midfield")
    assert avg > -0.05                                         # old rule: average midfielder (a better player)


def test_displaced_mass_is_monotone_and_moves_to_other_positions_when_needed():
    rest = M.without_player(squad(), 1, 99)
    order = M.displacement_order(rest, "Midfield")
    s = np.linspace(0, 1, 11)
    per = M.displaced_mass(order, s[1:]) / s[1:]
    assert np.all(np.diff(per) >= -1e-12)                      # the more he plays, the better the players he displaces
    big = M.displaced_mass(order, np.array([100.0]))[0]        # more than the whole squad: capped
    assert np.isclose(big, (rest.share * rest.rating).sum())


def test_delta_draws_weakest_is_positive_for_a_clear_upgrade():
    rest = M.without_player(squad(), 1, 99)
    order = M.displacement_order(rest, "Midfield")
    rng = np.random.default_rng(0)
    dl, r_inc = M.delta_draws_weakest(0.4, 0.01, 0.9, order, np.array([0.7]), np.array([0.2]), 2000, rng)
    assert (dl > 0).all() and r_inc < 0.4


def test_promoted_clubs_get_the_strength_of_the_clubs_they_replace():
    S = pd.Series({1: 0.5, 2: 0.1, 3: -0.3, 4: -0.5, 5: 9.9})     # 5 = stale squad from years ago
    out = M.promoted_strength(S, clubs_now=[1, 2, 5, 6], clubs_last=[1, 2, 3, 4])
    assert np.isclose(out[5], -0.4) and np.isclose(out[6], -0.4)
    assert out[1] == 0.5 and out[2] == 0.1
    assert S[5] == 9.9                                          # input not changed


def test_forecast_script_runs_on_synthetic_world():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "forecast_2026.py"), "--synthetic"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = ROOT / "reports"
    try:
        t = pd.read_csv(rep / "forecast_synthetic_table.csv")
        s = pd.read_csv(rep / "forecast_synthetic_signings.csv")
        c = pd.read_csv(rep / "lab_synthetic_curves.csv")
        for _, g in t.groupby("league"):
            n = len(g)
            assert np.isclose(g.p_title.sum(), 1.0, atol=0.02)
            assert np.isclose(g.p_top_half.sum(), n // 2, atol=0.05)
            assert sorted(g.pred_rank) == list(range(1, n + 1))
        assert s.regular.isna().all()                          # outcomes are unknown: a real forecast
        assert s.p_regular.between(0, 1).all()
        assert (c.groupby("club_id").points.apply(lambda x: np.all(np.diff(x.to_numpy()) > 0))).all()
    finally:
        for f in list(rep.glob("forecast_synthetic_*.csv")) + list(rep.glob("lab_synthetic_*.csv")):
            f.unlink()


def test_transfer_lab_estimate_upgrade_vs_downgrade():
    from freeagent import app_data as D
    sq = squad().assign(player_name=lambda d: "P" + d.player_id.astype(str))
    grid = np.linspace(-0.4, 0.4, 41)
    lab = {"squads": sq, "curves": pd.DataFrame({"club_id": 1, "delta": grid, "points": 50 + 40 * grid}),
           "players": pd.DataFrame({"player_id": [500, 501], "rating": [0.5, -0.3], "rating_sd": [0.01, 0.01],
                                    "position": ["Midfield", "Midfield"], "player_name": ["Star", "Dud"]})}
    star = D.lab_estimate(lab, 1, 500, 0.5, 10.0, 1.3)
    dud = D.lab_estimate(lab, 1, 501, 0.5, 10.0, 1.3)
    assert star["points_mean"] > 0 > dud["points_mean"]
    assert star["displaced"][0][0] == "P12"                    # the weakest midfielder loses his minutes first
    assert np.isclose(sum(s for _, s, _ in star["displaced"]), 0.5)
    assert abs(star["p_overpaid"] + star["p_fair"] + star["p_best_buy"] - 1) < 1e-9
