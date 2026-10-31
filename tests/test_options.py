"""Tests for the two extra prediction tasks: season outcome (Option A) and market-value rise (Option B).
The key tests are leakage tests: nothing dated on or after the cutoff may change the inputs."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeagent import data, simulate as M, success as S, value_rise as VR
from freeagent.impact import PlusMinus
from freeagent.synthetic import make_world, make_valuations

_W = {}


def world():
    if not _W:
        w = make_world(n_leagues=2, seasons=tuple(range(2014, 2020)), seed=3)
        w["player_valuations"] = make_valuations(w, seed=3)
        w = data.standardise(w)
        comps = sorted(w["games"].competition_id.unique())
        _W["w"], _W["d"] = w, data.prepare(w, competitions=comps, first_season=2014)
    return _W["w"], _W["d"]


def test_season_squad_uses_no_data_after_the_cutoffs():
    w, d = world()
    pm = PlusMinus(lam=20, halflife_days=730).fit(d["games"], d["stints"], "2018-08-01")
    full = M.season_squad(d["games"], d["stints"], pm.ratings, w["players"], w["transfers"], 2018, new_share=0.4)
    g_cut = d["games"][d["games"].date < pd.Timestamp("2018-07-01")]
    s_cut = d["stints"][d["stints"].game_id.isin(g_cut.game_id)]
    t_cut = w["transfers"][w["transfers"].transfer_date < pd.Timestamp("2018-08-01")]
    trunc = M.season_squad(g_cut, s_cut, pm.ratings, w["players"], t_cut, 2018, new_share=0.4)
    a = M.club_strength(full).sort_index()
    b = M.club_strength(trunc).sort_index()
    assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy())


def test_season_squad_keeps_eleven_players_worth_of_minutes():
    w, d = world()
    pm = PlusMinus(lam=20, halflife_days=730).fit(d["games"], d["stints"], "2018-08-01")
    base = M.baseline_squad(d["games"], d["stints"], pm.ratings, w["players"], 2018)
    new = M.season_squad(d["games"], d["stints"], pm.ratings, w["players"], w["transfers"], 2018, new_share=0.4)
    tb, tn = base.groupby("club_id").share.sum(), new.groupby("club_id").share.sum()
    assert np.allclose(tb.to_numpy(), tn.reindex(tb.index).to_numpy())


def test_simulated_top_half_probabilities_are_valid():
    w, d = world()
    pm = PlusMinus(lam=20, halflife_days=730).fit(d["games"], d["stints"], "2018-08-01")
    sq = M.baseline_squad(d["games"], d["stints"], pm.ratings, w["players"], 2018)
    S_ = M.club_strength(sq)

    class Fake:
        coef, rho = (0.25, 0.15, 0.35), 0.0
    lg = d["games"].competition_id.iloc[0]
    fx = d["games"][(d["games"].season == 2018) & (d["games"].competition_id == lg)]
    t = M.simulate_table(Fake(), fx, S_, n_sims=300)
    assert t.p_top_half.between(0, 1).all()
    assert abs(t.p_top_half.sum() - len(t) // 2) < 1e-9          # exactly half the league is top half in every run


def test_value_rise_features_ignore_future_valuations():
    w, d = world()
    lt = S.league_tables(w)
    full = VR.build_panel(w, lt, [2018])
    w2 = dict(w)
    v = w["player_valuations"].copy()
    future = v.date >= pd.Timestamp("2018-08-01")
    v.loc[future, "market_value_in_eur"] *= np.random.default_rng(0).uniform(0.5, 2.0, future.sum())
    w2["player_valuations"] = v
    pert = VR.build_panel(w2, lt, [2018])
    feats = [f for f in VR.NUMERIC if full[f].notna().any()]
    m = full.merge(pert, on=["player_id", "season"], suffixes=("", "_p"))
    for f in feats:
        assert np.allclose(m[f].to_numpy(float), m[f + "_p"].to_numpy(float), equal_nan=True), f
    assert (m.rise != m.rise_p).any()                             # the label DOES depend on future values


def test_option_scripts_run_on_synthetic_world():
    for script in ("season_predict.py", "value_rise.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--synthetic"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (script, r.stderr[-800:])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "season_points_test.py"), "--boot", "200",
                        "--file", str(ROOT / "reports" / "season_synthetic_predictions.csv")], capture_output=True, text=True)
    assert r.returncode == 0 and "Rank correlation" in r.stdout, r.stderr[-800:]
    for f in (ROOT / "reports").glob("*synthetic*.csv"):
        f.unlink()
