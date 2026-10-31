"""Tests for M2 (ML prior) and M8 (signing-success classifier).

The most important one is the leakage test: features computed with ALL data must equal
features computed after deleting every row dated on or after the transfer date.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent.synthetic import make_world
from freeagent import data, success as S
from freeagent.impact import PlusMinus
from freeagent.prior import Prior, player_stats

_W = {}


def world():
    if not _W:
        w = data.standardise(make_world(n_leagues=2, seasons=tuple(range(2014, 2020)), seed=3))
        comps = sorted(w["games"].competition_id.unique())
        _W["w"], _W["d"] = w, data.prepare(w, competitions=comps, first_season=2014)
    return _W["w"], _W["d"]


# ---------- M8 ----------
def test_signing_labels_are_consistent():
    w, _ = world()
    sig = S.build_signings(w)
    assert len(sig) > 50
    assert sig.share.between(0, 1).all()
    assert ((sig.share >= 0.5) == (sig.regular == 1)).all()
    assert (sig.transfer_fee >= 5e6).all()


def test_features_use_no_data_from_on_or_after_transfer_date():
    w, _ = world()
    sig = S.build_signings(w)
    s = sig[sig.season == 2017]
    T = s.transfer_date.min()
    full = S.build_features(s, w)
    cut = {k: v.copy() for k, v in w.items()}
    cut["games"] = w["games"][w["games"].date < T]
    cut["appearances"] = w["appearances"][w["appearances"].date < T]
    cut["transfers"] = w["transfers"][w["transfers"].transfer_date < T]
    trunc = S.build_features(s, cut)
    for f in S.NUMERIC + S.VALUE_FEATURES:
        a, b = full[f].to_numpy(float), trunc[f].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"feature {f} changes when future data is removed"


def test_classifier_beats_base_rate_on_synthetic_world():
    w, d = world()
    r = S.fit_ratings(d["games"], d["stints"], range(2015, 2020), lam=20, halflife_days=730)
    df = S.build_features(S.build_signings(w), w, ratings=r)
    # the synthetic world has no market values, so drop columns that are entirely empty
    feats = [f for f in S.NUMERIC + S.CATEGORICAL if df[f].notna().any()]
    assert "rating" in feats
    pr =S.walk_forward(df, lambda: S.Ensemble(feats, max_depth=2, learning_rate=0.05, max_iter=100),
                        feats, [2018, 2019])
    ours = S.scores(pr.p.to_numpy(), pr.regular.to_numpy())
    base = max(pr.regular.mean(), 1 - pr.regular.mean())
    assert ours["auc"] > 0.6 and ours["accuracy"] > base - 0.02


# ---------- M2 ----------
def test_prior_stats_use_only_earlier_matches():
    w, d = world()
    cut = pd.Timestamp("2017-08-01")
    a = player_stats(d["appearances"], w["players"], cut)
    b = player_stats(d["appearances"][d["appearances"].date < cut], w["players"], cut)
    assert np.allclose(a.drop(columns="position").to_numpy(float), b.drop(columns="position").to_numpy(float),
                       equal_nan=True)


def test_prior_helps_rate_players_with_few_minutes():
    w, d = world()
    cut = "2019-01-01"
    pr = Prior(lam=20, halflife_days=730).fit(d["games"], d["stints"], d["appearances"], w["players"], cut)
    with_p = pr.ratings(d["games"], d["stints"], cut)
    no_p = pr.base.ratings
    truth = w["truth"].set_index("player_id").theta
    few = pr.base.minutes[pr.base.minutes < 1500].index
    assert len(few) > 30
    assert with_p[few].corr(truth[few]) > no_p[few].corr(truth[few])


# ---------- M8 round 2 ----------
def test_extra_features_use_no_data_from_on_or_after_transfer_date():
    w, d = world()
    r = S.fit_ratings(d["games"], d["stints"], range(2015, 2020), lam=20, halflife_days=730)
    sig = S.build_signings(w)
    s = sig[sig.season == 2017]
    T = s.transfer_date.min()
    full = S.add_extra_features(S.build_features(s, w, ratings=r), w, ratings=r)
    cut = {k: v.copy() for k, v in w.items()}
    cut["games"] = w["games"][w["games"].date < T]
    cut["appearances"] = w["appearances"][w["appearances"].date < T]
    cut["transfers"] = w["transfers"][w["transfers"].transfer_date < T]
    trunc = S.add_extra_features(S.build_features(s, cut, ratings=r), cut, ratings=r)
    for f in S.EXTRA:
        a, b = full[f].to_numpy(float), trunc[f].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"feature {f} changes when future data is removed"
    assert full.rating_gap.notna().any() and full.club_euro.notna().any()


def test_published_model_families_run():
    w, d = world()
    r = S.fit_ratings(d["games"], d["stints"], range(2015, 2020), lam=20, halflife_days=730)
    df = S.add_extra_features(S.build_features(S.build_signings(w), w, ratings=r), w, ratings=r)
    feats = [f for f in S.NUMERIC + S.CATEGORICAL + S.EXTRA if df[f].notna().any()]
    for kind in ("rf", "ada", "nb"):
        pr = S.walk_forward(df, lambda: S.make_baseline(kind, feats), feats, [2019])
        assert pr.p.between(0, 1).all() and len(pr) > 0


def test_comparison_panel_models_run():
    w, d = world()
    r = S.fit_ratings(d["games"], d["stints"], range(2015, 2020), lam=20, halflife_days=730)
    df = S.build_features(S.build_signings(w), w, ratings=r)
    feats = [f for f in S.NUMERIC + S.CATEGORICAL if df[f].notna().any()]
    for kind in ("rf", "et", "ada", "nb", "knn", "svm", "mlp"):
        pr = S.walk_forward(df, lambda: S.make_panel_model(kind, feats), feats, [2019])
        assert pr.p.between(0, 1).all() and len(pr) > 0, kind


def test_robustness_script_runs_on_synthetic_world():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "robustness_checks.py"), "--synthetic",
                        "--parts", "threshold", "position", "halflife_m8"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "Rating half-life" in r.stdout and "By position" in r.stdout
    for f in (root / "reports").glob("robustness_synthetic_*.csv"):
        f.unlink()
