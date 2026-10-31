"""Tests for M5 (season simulator)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent.synthetic import make_world
from freeagent import data, simulate as M
from freeagent.impact import PlusMinus
from freeagent.match import PlayerPoisson, honest_strength

_W = {}


def setup():
    if not _W:
        w = data.standardise(make_world(n_leagues=1, clubs_per_league=10, seasons=tuple(range(2014, 2019)), seed=5))
        d = data.prepare(w, competitions=["GB1"], first_season=2014)
        G, S = d["games"], d["stints"]
        rf = lambda cut: PlusMinus(lam=20, halflife_days=730).fit(G, S, cut).ratings
        st = honest_strength(G, S, [2015, 2016, 2017], rf)
        g = G.merge(st, on="game_id", how="left")
        model = PlayerPoisson().fit(g, pd.Timestamp("2018-08-01"))
        cut = pd.Timestamp("2018-08-01")
        ratings = rf(cut)
        sq = M.baseline_squad(G, S, ratings, w["players"], 2018)
        fx = G[G.season == 2018]
        _W.update(model=model, sq=sq, fx=fx, S=M.club_strength(sq))
    return _W


def test_delta_zero_means_zero_points_gain():
    W = setup()
    club = int(W["fx"].home_club_id.iloc[0])
    gain = M.points_gain(W["model"], W["fx"], W["S"], club, np.zeros(5))
    assert np.allclose(gain, 0)


def test_better_signing_adds_more_points_and_curve_is_monotone():
    W = setup()
    club = int(W["fx"].home_club_id.iloc[0])
    grid = np.linspace(-0.5, 0.5, 11)
    curve = M.points_curve(W["model"], W["fx"], W["S"], club, grid)
    assert np.all(np.diff(curve) > 0)
    assert M.points_gain(W["model"], W["fx"], W["S"], club, np.array([0.3]))[0] > 0
    assert M.points_gain(W["model"], W["fx"], W["S"], club, np.array([-0.3]))[0] < 0


def test_simulation_agrees_with_exact_expected_points():
    W = setup()
    club = int(W["fx"].home_club_id.iloc[0])
    exact = M.points_curve(W["model"], W["fx"], W["S"], club, np.array([0.0]))[0]
    sim = M.simulate_table(W["model"], W["fx"], W["S"], n_sims=4000).loc[club, "mean_points"]
    assert abs(sim - exact) < 0.6          # Monte Carlo error on a 36-match season is about 0.15


def test_common_random_numbers_give_zero_difference_without_signing_and_positive_with():
    W = setup()
    club = int(W["fx"].home_club_id.iloc[0])
    base = M.simulate_table(W["model"], W["fx"], W["S"], n_sims=1000)
    same = M.simulate_table(W["model"], W["fx"], W["S"], shifts={club: 0.0}, n_sims=1000)
    better = M.simulate_table(W["model"], W["fx"], W["S"], shifts={club: 0.4}, n_sims=1000)
    assert np.allclose(base.mean_points, same.mean_points)
    assert better.loc[club, "mean_points"] > base.loc[club, "mean_points"]
    assert better.loc[club, "p_top4"] >= base.loc[club, "p_top4"]


def test_delta_draws_displacement_logic():
    rng = np.random.default_rng(0)
    pool = np.array([0.6])
    # replacing someone equally good changes nothing; a better player helps; a worse one hurts
    same = M.delta_draws(0.2, 1e-9, 1.0, 0.2, pool, pool, 100, rng)
    better = M.delta_draws(0.5, 1e-9, 1.0, 0.2, pool, pool, 100, rng)
    worse = M.delta_draws(-0.1, 1e-9, 1.0, 0.2, pool, pool, 100, rng)
    assert np.allclose(same, 0, atol=1e-6) and better.mean() > 0 > worse.mean()
    # a bench player (small share) moves the needle less than a regular
    bench = M.delta_draws(0.5, 1e-9, 0.0, 0.2, np.array([0.7]), np.array([0.1]), 100, rng)
    assert bench.mean() < better.mean()
