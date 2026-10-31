"""RQ1 go/no-go: does the player-based match model beat team-only models, using only
pre-match information? Walk-forward: for each season, train on everything before
1 August, predict that season's top-5-league matches.

Usage:
  python scripts/rq1_match_backtest.py                 # real data: data/freeagent_subset.zip
  python scripts/rq1_match_backtest.py --synthetic     # fake world, for a quick check
Outputs: reports/rq1_match_models.csv (per season) and reports/rq1_summary.csv (pooled).
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

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--synthetic", action="store_true")
p.add_argument("--seasons", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
args = p.parse_args()

t0 = time.time()
if args.synthetic:
    from freeagent.synthetic import make_world
    tables = make_world(n_leagues=4, seasons=tuple(range(2014, 2022)), seed=1)
    comps = sorted(tables["games"].competition_id.unique())
    eval_comps, seasons = comps, [2018, 2019, 2020, 2021]
else:
    tables = data.load_tables(args.data)
    comps, eval_comps, seasons = config.RATING_COMPETITIONS, config.TOP_LEAGUES, args.seasons

d = data.prepare(tables, competitions=comps)
G, S = d["games"], d["stints"]
print(f"Loaded {len(G):,} games, {len(S):,} player-match rows in {time.time() - t0:.0f}s")
print(d["coverage"].to_string())

elo = match.elo_gaps(G)          # pre-match Elo gap for every game (uses only earlier results)


def ratings_fn(lam_, hl_):
    return lambda cut: PlusMinus(lam=lam_, halflife_days=hl_).fit(G, S, cut).ratings


def with_strength(lam_, hl_, seasons_):
    st = match.honest_strength(G, S, seasons_, ratings_fn(lam_, hl_))
    return G.merge(st, on="game_id", how="left").merge(elo, on="game_id")


def season_slice(g_, s_):
    c, e = pd.Timestamp(f"{s_}-08-01"), pd.Timestamp(f"{s_ + 1}-07-31")
    return g_[(g_.date >= c) & (g_.date <= e) & g_.competition_id.isin(eval_comps)].dropna(subset=["S_home", "S_away"])


# Tune (lam, halflife) on the season BEFORE the first test season, by pre-match log loss.
# Match-model coefficients are trained on cross-fitted strengths from the 3 seasons before it.
first = seasons[0]
grid = []
for lam_ in (30, 100, 300):
    for hl_ in (730, 1460):
        g_ = with_strength(lam_, hl_, range(first - 4, first))
        v = season_slice(g_, first - 1)
        prob = match.PlayerPoisson().fit(g_, pd.Timestamp(f"{first - 1}-08-01")).predict(v)
        grid.append({"lam": lam_, "halflife_days": hl_,
                     "valid_log_loss": match.metrics(prob, match.outcome_label(v))["log_loss"]})
grid = pd.DataFrame(grid).sort_values("valid_log_loss").reset_index(drop=True)
print("\nRating settings (validation log loss, season before testing):\n", grid.to_string())
lam, hl = float(grid.lam[0]), float(grid.halflife_days[0])

g = with_strength(lam, hl, range(first - 3, seasons[-1] + 1))
rows = []
for season in seasons:
    cut = pd.Timestamp(f"{season}-08-01")
    t = time.time()
    test = season_slice(g, season)
    y = match.outcome_label(test)
    for name, model in [("Frequency", match.Frequency()), ("Elo", match.Elo()),
                        ("Team Dixon-Coles", match.TeamDixonColes()),
                        ("Player model (ours)", match.PlayerPoisson()),
                        ("Hybrid: player + Elo (ours)", match.HybridPoisson())]:
        prob = model.fit(g, cut).predict(test)
        rows.append({"season": season, "model": name, **match.metrics(prob, y)})
        rows[-1]["_p"], rows[-1]["_y"] = prob, y
    print(f"Season {season}: {len(test)} test matches, {time.time() - t:.0f}s")

res = pd.DataFrame(rows)
out = config.REPORTS
out.mkdir(exist_ok=True)
res.drop(columns=["_p", "_y"]).to_csv(out / "rq1_match_models.csv", index=False)
pooled = {m: match.metrics(np.vstack(v._p.tolist()), np.concatenate(v._y.tolist())) for m, v in res.groupby("model")}
summary = pd.DataFrame(pooled).T.sort_values("log_loss")
summary.to_csv(out / "rq1_summary.csv")

# Paired bootstrap of the log-loss difference between two models (negative = first is better).
def compare(a, b):
    A, B = res[res.model == a], res[res.model == b]
    y_all = np.concatenate(A._y.tolist())
    ll = lambda P: -np.log(np.clip(np.vstack(P.tolist())[np.arange(len(y_all)), y_all], 1e-12, 1))
    diff = ll(A._p) - ll(B._p)
    rng = np.random.default_rng(config.SEED)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "significantly better" if hi < 0 else "significantly worse" if lo > 0 else "no significant difference"
    return f"{a} vs {b}: {diff.mean():+.4f} (95% CI {lo:+.4f} to {hi:+.4f}) -> {verdict}"


print("\n=== RQ1 pooled results (lower log loss / RPS / Brier is better) ===")
print(summary.round(4).to_string())
print()
for a, b in [("Player model (ours)", "Team Dixon-Coles"), ("Hybrid: player + Elo (ours)", "Elo"),
             ("Hybrid: player + Elo (ours)", "Team Dixon-Coles")]:
    print(compare(a, b))
print(f"Total time {time.time() - t0:.0f}s. Saved reports/rq1_match_models.csv and rq1_summary.csv")
