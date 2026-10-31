"""M2 check: does the ML prior improve the ratings?

Two tests, both strictly out of sample:
  A. Prior quality. At cutoff 1 Aug Y, take players with little history (< 900 rated minutes).
     Compare their prior mu (and their no-prior rating) with the rating they have three
     years later, once they have played >= 2000 more minutes. Higher correlation = better guess.
  B. Match prediction (RQ1 again). Same walk-forward as rq1_match_backtest.py with the
     rating settings it chose (lam = 100, halflife = 1460 days), with and without the prior.

Usage:  python scripts/m2_prior_backtest.py
Outputs: reports/m2_prior_quality.csv, reports/m2_match_models.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, match
from freeagent.impact import PlusMinus
from freeagent.prior import Prior

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--seasons", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
p.add_argument("--lam", type=float, default=100.0)
p.add_argument("--halflife", type=float, default=1460.0)
args = p.parse_args()
t0 = time.time()
LAM, HL = args.lam, args.halflife

tables = data.load_tables(args.data)
d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
G, S, A, P = d["games"], d["stints"], d["appearances"], tables["players"]
print(f"Loaded {len(G):,} games [{time.time() - t0:.0f}s]")

# ---- A. prior quality for players with little history
rows = []
for y in (2017, 2018, 2019, 2020, 2021):
    cut, later = f"{y}-08-01", f"{y + 3}-08-01"
    pr = Prior(LAM, HL).fit(G, S, A, P, cut)
    fut = PlusMinus(lam=LAM, halflife_days=HL).fit(G, S, later)
    new = pd.DataFrame({"mins_then": pr.base.minutes, "no_prior": pr.base.ratings}).reindex(pr.mu.index)
    new["mins_then"] = new.mins_then.fillna(0)
    new["prior"] = pr.mu
    with_prior = PlusMinus(lam=LAM, halflife_days=HL).fit(G, S, cut, prior=pr.mu).ratings
    new["with_prior"] = with_prior.reindex(new.index).fillna(new.prior)
    new["no_prior"] = new.no_prior.fillna(0)
    new["future"] = fut.ratings.reindex(new.index)
    new["future_mins"] = fut.minutes.reindex(new.index).fillna(0) - new.mins_then
    sel = new[(new.mins_then < 900) & (new.future_mins >= 2000)].dropna(subset=["future"])
    rows.append({"cutoff": cut, "players": len(sel), "prior_targets": pr.n_targets,
                 "corr_no_prior": sel.no_prior.corr(sel.future),
                 "corr_with_prior": sel.with_prior.corr(sel.future),
                 "corr_prior_alone": sel.prior.corr(sel.future)})
    print(f"  prior quality {cut}: {rows[-1]['players']} new players [{time.time() - t0:.0f}s]")
qual = pd.DataFrame(rows)
config.REPORTS.mkdir(exist_ok=True)
qual.to_csv(config.REPORTS / "m2_prior_quality.csv", index=False)
print("\nA. How well do we rate players with < 900 minutes? (correlation with their rating 3 years later)")
print(qual.round(3).to_string(index=False))

# ---- B. match prediction with and without the prior
elo = match.elo_gaps(G)
first = args.seasons[0]


def no_prior(cut):
    return PlusMinus(lam=LAM, halflife_days=HL).fit(G, S, cut).ratings


def with_prior(cut):
    pr = Prior(LAM, HL).fit(G, S, A, P, cut)
    return pr.ratings(G, S, cut).combine_first(pr.mu)


def strengths(fn):
    st = match.honest_strength(G, S, range(first - 3, args.seasons[-1] + 1), fn)
    return G.merge(st, on="game_id", how="left").merge(elo, on="game_id")


g_no, g_pr = strengths(no_prior), strengths(with_prior)
print(f"\nStrengths computed [{time.time() - t0:.0f}s]")
res = []
for season in args.seasons:
    cut, end = pd.Timestamp(f"{season}-08-01"), pd.Timestamp(f"{season + 1}-07-31")
    for tag, g in [("no prior", g_no), ("with prior", g_pr)]:
        test = g[(g.date >= cut) & (g.date <= end) & g.competition_id.isin(config.TOP_LEAGUES)]
        test = test.dropna(subset=["S_home", "S_away"])
        for name, model in [("Player model", match.PlayerPoisson()), ("Hybrid: player + Elo", match.HybridPoisson())]:
            prob = model.fit(g, cut).predict(test)
            res.append({"season": season, "model": f"{name} ({tag})", "_p": prob,
                        "_y": match.outcome_label(test), "_id": test.game_id.to_numpy()})
res = pd.DataFrame(res)

# compare on the same matches only
common = set.intersection(*[set(np.concatenate(v._id.tolist())) for _, v in res.groupby("model")])
pooled, LL = {}, {}
for m, v in res.groupby("model"):
    ids, P_, Y_ = (np.concatenate(v[c].tolist()) for c in ("_id", "_p", "_y"))
    keep = np.isin(ids, list(common))
    order = np.argsort(ids[keep])
    P_, Y_ = np.vstack(v._p.tolist())[keep][order], Y_[keep][order]
    pooled[m] = match.metrics(P_, Y_)
    LL[m] = -np.log(np.clip(P_[np.arange(len(Y_)), Y_], 1e-12, 1))
summary = pd.DataFrame(pooled).T.sort_values("log_loss")
summary.to_csv(config.REPORTS / "m2_match_models.csv")
print("\nB. Match prediction, same walk-forward as RQ1 (lower log loss is better)")
print(summary.round(4).to_string())
rng = np.random.default_rng(config.SEED)
for name in ("Player model", "Hybrid: player + Elo"):
    diff = LL[f"{name} (with prior)"] - LL[f"{name} (no prior)"]
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    v = "significantly better" if hi < 0 else "significantly worse" if lo > 0 else "no significant difference"
    print(f"{name}: with prior vs no prior {diff.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}) -> {v}")
print(f"Total time {time.time() - t0:.0f}s. Saved reports/m2_*.csv")
