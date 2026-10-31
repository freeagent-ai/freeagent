"""Are our season POINTS forecasts significantly better than 'same points as last season'?

Reads the saved predictions of season_predict.py (no new data, nothing re-fitted, nothing tuned).
Compared on the clubs that were in the same league last season (promoted clubs have no last-season points).
  1. Mean absolute error of final points (lower is better).
  2. Rank correlation with the final table (Spearman, one per league-season; higher is better).
Two bootstraps:
  * club-level: resample club-seasons (standard paired bootstrap);
  * league-season level (STRICTER): resample whole league-seasons, because clubs in the same table are not
    independent. The verdict we quote is the stricter one.
'Last season's points' are scaled to this season's number of matches (done in season_predict.py).

Usage: python scripts/season_points_test.py [--file reports/season_predictions.csv]
Output: reports/season_points_test.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config

p = argparse.ArgumentParser()
p.add_argument("--file", default=str(config.REPORTS / "season_predictions.csv"))
p.add_argument("--boot", type=int, default=5000)
args = p.parse_args()
rng = np.random.default_rng(config.SEED)
OUT = config.REPORTS / ("season_points_test" + ("_synthetic" if "synthetic" in Path(args.file).name else "") + ".csv")

res = pd.read_csv(args.file)
d = res[res.lastseason_points.notna()].copy()
d["err_ours"] = (d.pred_points - d.actual_points).abs()
d["err_last"] = (d.lastseason_points - d.actual_points).abs()
d["group"] = d.season.astype(str) + "-" + d.league.astype(str)
print(f"{len(d)} club-seasons with last-season points in the same league "
      f"({len(res) - len(d)} promoted clubs left out), {d.group.nunique()} league-seasons")


def verdict(lo, hi):
    return "OURS SIGNIFICANTLY BETTER" if lo > 0 else "ours significantly worse" if hi < 0 else "tied (not significant)"


rows = []
# ---- 1. mean absolute error, positive gain = ours has the smaller error
gain = (d.err_last - d.err_ours).to_numpy()
b = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(args.boot)]
lo1, hi1 = np.percentile(b, [2.5, 97.5])
groups = {g: x.index.to_numpy() for g, x in d.groupby("group")}
keys = list(groups)
pos = pd.Series(np.arange(len(d)), index=d.index)
bc = []
for _ in range(args.boot):
    pick = rng.choice(keys, len(keys))
    idx = np.concatenate([pos[groups[k]].to_numpy() for k in pick])
    bc.append(gain[idx].mean())
lo2, hi2 = np.percentile(bc, [2.5, 97.5])
rows.append({"measure": "mean absolute error of final points", "ours": d.err_ours.mean(), "last_season": d.err_last.mean(),
             "ours_advantage": gain.mean(), "club_ci_lo": lo1, "club_ci_hi": hi1, "club_verdict": verdict(lo1, hi1),
             "league_season_ci_lo": lo2, "league_season_ci_hi": hi2, "league_season_verdict": verdict(lo2, hi2)})

# ---- 2. rank correlation per league-season, positive = ours ranks the table better
rho = d.groupby("group").apply(lambda g: pd.Series({
    "ours": spearmanr(g.pred_points, g.actual_points)[0],
    "last": spearmanr(g.lastseason_points, g.actual_points)[0]}), include_groups=False).dropna()
dr = (rho.ours - rho["last"]).to_numpy()
bs = [dr[rng.integers(0, len(dr), len(dr))].mean() for _ in range(args.boot)]
lo3, hi3 = np.percentile(bs, [2.5, 97.5])
rows.append({"measure": "rank correlation with final table (Spearman)", "ours": rho.ours.mean(),
             "last_season": rho["last"].mean(), "ours_advantage": dr.mean(),
             "league_season_ci_lo": lo3, "league_season_ci_hi": hi3, "league_season_verdict": verdict(lo3, hi3),
             "ours_wins_in": f"{int((dr > 0).sum())} of {len(dr)} league-seasons"})
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("\n=== 1. Final points: mean absolute error (lower is better) ===")
print(f"  ours {rows[0]['ours']:.2f}  vs  same as last season {rows[0]['last_season']:.2f}   "
      f"-> ours better by {gain.mean():.2f} points per club")
print(f"  club-level bootstrap:         95% CI {lo1:+.2f} to {hi1:+.2f}  -> {verdict(lo1, hi1)}")
print(f"  league-season bootstrap:      95% CI {lo2:+.2f} to {hi2:+.2f}  -> {verdict(lo2, hi2)}   (stricter, quote this)")
print(f"  ours has the smaller error for {np.mean(gain > 0):.1%} of clubs")
print("\n=== 2. Rank correlation with the final table (higher is better) ===")
print(f"  ours {rho.ours.mean():.3f}  vs  last season's points {rho['last'].mean():.3f}  "
      f"-> ours better in {int((dr > 0).sum())} of {len(dr)} league-seasons")
print(f"  league-season bootstrap:      95% CI {lo3:+.3f} to {hi3:+.3f}  -> {verdict(lo3, hi3)}")
print(f"\nSaved {OUT.relative_to(config.REPORTS.parent)}")
