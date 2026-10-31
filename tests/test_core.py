"""Tests for M1, M3, M4. Run with `pytest` or `python tests/run_tests.py`.

The synthetic world has KNOWN true player ratings, so these tests check that the
maths actually recovers the truth, not just that the code runs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent.synthetic import make_world
from freeagent import data, match
from freeagent.impact import PlusMinus

_CACHE = {}


def world(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        w = make_world(**kw)
        comps = sorted(w["games"].competition_id.unique())
        _CACHE[key] = (w, data.prepare(w, competitions=comps, first_season=2014))
    return _CACHE[key]


# ---------- M1 ----------
def test_synthetic_matches_kaggle_schema():
    w, _ = world()
    assert {"game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
            "home_club_goals", "away_club_goals"} <= set(w["games"].columns)
    assert {"game_id", "player_id", "player_club_id", "minutes_played"} <= set(w["appearances"].columns)


def test_stints_have_eleven_players_worth_of_minutes_per_side():
    _, d = world()
    per_side = d["stints"].groupby(["game_id", "sign"]).share.sum()
    assert np.allclose(per_side, 11, atol=0.05)


def test_incomplete_lineups_are_excluded_but_result_kept():
    w, _ = world()
    w2 = dict(w)
    first = w["games"].game_id.iloc[0]
    w2["appearances"] = w["appearances"][~((w["appearances"].game_id == first) &
                                           (w["appearances"].player_club_id == w["games"].home_club_id.iloc[0]))]
    d = data.prepare(w2, competitions=["GB1", "ES1"], first_season=2014)
    assert first in set(d["games"].game_id)
    assert not d["games"].set_index("game_id").has_lineups[first]
    assert first not in set(d["stints"].game_id)


# ---------- M3 ----------
def _corr(model, w):
    truth = w["truth"].set_index("player_id").theta
    both = pd.concat([model.ratings, truth], axis=1, join="inner")
    return both.corr().iloc[0, 1]


def test_multi_season_ratings_beat_single_season():
    w, d = world()
    multi = PlusMinus(lam=80, halflife_days=1460).fit(d["games"], d["stints"])
    single = PlusMinus(lam=80).fit(d["games"][d["games"].season == 2014], d["stints"])
    assert _corr(multi, w) > 0.3
    assert _corr(multi, w) > _corr(single, w) + 0.15


def test_strong_prior_pulls_ratings_to_prior():
    w, d = world()
    truth = w["truth"].set_index("player_id").theta
    m = PlusMinus(lam=1e6).fit(d["games"], d["stints"], prior=truth)
    assert np.allclose(m.ratings, truth.reindex(m.ratings.index), atol=1e-3)


def test_good_prior_improves_recovery():
    w, d = world()
    truth = w["truth"].set_index("player_id").theta
    noisy = truth + np.random.default_rng(0).normal(0, 0.15, len(truth))
    plain = PlusMinus(lam=80).fit(d["games"], d["stints"])
    informed = PlusMinus(lam=80).fit(d["games"], d["stints"], prior=noisy)
    assert _corr(informed, w) > _corr(plain, w)


def test_posterior_covariance_is_valid():
    _, d = world()
    m = PlusMinus(lam=20).fit(d["games"], d["stints"])
    c = m.covariance(list(m.ratings.index[:5])).to_numpy()
    assert np.allclose(c, c.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(c) > 0)


def test_fit_ignores_matches_after_cutoff():
    _, d = world()
    m = PlusMinus().fit(d["games"], d["stints"], cutoff="2016-08-01")
    assert m.n_games == int((d["games"].date < "2016-08-01").sum())


# ---------- M4 ----------
def test_probabilities_sum_to_one_and_tau_neutral_at_zero():
    p = match.outcome_probs(np.array([1.4, 0.6, 2.5]), np.array([1.1, 1.9, 0.4]), rho=-0.08)
    assert np.allclose(p.sum(1), 1)
    assert np.allclose(match.tau(np.array([0, 1]), np.array([0, 1]), 1.3, 1.1, 0.0), 1)


def test_dixon_coles_finds_home_advantage():
    _, d = world()
    m = match.TeamDixonColes().fit(d["games"], "2019-08-01")
    assert m.coef[1] > 0


def test_expected_strength_uses_only_earlier_matches():
    _, d = world()
    G, S = d["games"], d["stints"]
    r = pd.Series(1.0, index=S.player_id.unique())
    base = match.expected_strength(G, S, r).set_index("game_id")
    gid = G.game_id.iloc[200]
    S2 = S.copy(); S2.loc[S2.game_id == gid, "share"] *= 5          # tamper with that match's own line-up
    after = match.expected_strength(G, S2, r).set_index("game_id")
    assert np.isclose(base.loc[gid, "S_home"], after.loc[gid, "S_home"])


def test_player_model_beats_frequency_out_of_sample():
    w, d = world(seasons=tuple(range(2014, 2020)))
    G, S = d["games"], d["stints"]
    cut = "2019-08-01"
    pm = PlusMinus(lam=80, halflife_days=1460).fit(G, S, cut)
    g = G.merge(match.expected_strength(G, S, pm.ratings), on="game_id")
    test = g[g.date >= cut]
    y = match.outcome_label(test)
    ours = match.metrics(match.PlayerPoisson().fit(g, cut).predict(test), y)
    base = match.metrics(match.Frequency().fit(g, cut).predict(test), y)
    assert ours["log_loss"] < base["log_loss"]
