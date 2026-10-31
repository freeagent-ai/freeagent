"""M7: do our predictions match what really happened?

Reality measure that does NOT reuse our rating model: for each signing, his club's goal difference
per league match in his first season when he played >= 45 minutes minus when he did not
("on/off"). Only players with >= 8 matches on and >= 8 off are scored. On/off is noisy (rotation,
injuries, fixture difficulty), so we compare GROUPS of signings, with bootstrap intervals,
never single players.

Tests:
  1. Predicted points gain (M5): top third vs bottom third of signings -> realised on/off.
  2. Verdict groups (M6): Best Buy / Fair / Overpaid -> realised on/off.
  3. Rank correlation (Spearman) between predicted points and realised on/off.
Input: reports/m5_signing_points.csv.   Output: reports/m7_validation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data

tables = data.load_tables(str(config.DATA / "freeagent_subset.zip"))
G = tables["games"]
G = G[G.competition_id.isin(config.TOP_LEAGUES)].dropna(subset=["home_club_goals"])
A = tables["appearances"][["game_id", "player_id", "player_club_id", "minutes_played"]]
o = pd.read_csv(config.REPORTS / "m5_signing_points.csv")

side = pd.concat([
    G[["game_id", "season", "competition_id", "home_club_id", "home_club_goals", "away_club_goals"]]
    .set_axis(["game_id", "season", "league", "club_id", "gf", "ga"], axis=1),
    G[["game_id", "season", "competition_id", "away_club_id", "away_club_goals", "home_club_goals"]]
    .set_axis(["game_id", "season", "league", "club_id", "gf", "ga"], axis=1)])
side["gd"] = side.gf - side.ga
mins = A.groupby(["game_id", "player_id", "player_club_id"]).minutes_played.sum().reset_index()

rows = []
for r in o.itertuples():
    g = side[(side.club_id == r.to_club_id) & (side.season == r.season)][["game_id", "gd"]]
    m = mins[(mins.player_id == r.player_id) & (mins.player_club_id == r.to_club_id)][["game_id", "minutes_played"]]
    g = g.merge(m, on="game_id", how="left").fillna({"minutes_played": 0})
    on = g[g.minutes_played >= 45].gd
    off = g[g.minutes_played < 45].gd
    rows.append({"idx": r.Index, "n_on": len(on), "n_off": len(off),
                 "on_off": on.mean() - off.mean() if len(on) >= 8 and len(off) >= 8 else np.nan})
oo = o.join(pd.DataFrame(rows).set_index("idx"))
s = oo.dropna(subset=["on_off"]).copy()
print(f"{len(s)} of {len(o)} signings have enough matches on and off to be scored")
print(f"Average realised on/off goal difference per match: {s.on_off.mean():+.3f}\n")

rng = np.random.default_rng(config.SEED)


def boot_mean(x, n=3000):
    x = np.asarray(x)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def boot_diff(a, b, n=3000):
    a, b = np.asarray(a), np.asarray(b)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


out = []
s["tercile"] = pd.qcut(s.points_mean, 3, labels=["Bottom third", "Middle third", "Top third"])
print("1. By predicted points gain (M5)")
for k, g in s.groupby("tercile", observed=True):
    lo, hi = boot_mean(g.on_off)
    print(f"   {k:13s} n={len(g):4d}  predicted {g.points_mean.mean():+.2f} pts   realised on/off {g.on_off.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
    out.append({"test": "predicted tercile", "group": k, "n": len(g), "predicted_points": g.points_mean.mean(),
                "realised_on_off": g.on_off.mean(), "ci_lo": lo, "ci_hi": hi})
top, bot = s[s.tercile == "Top third"].on_off, s[s.tercile == "Bottom third"].on_off
lo, hi = boot_diff(top, bot)
verdict = "significant" if lo > 0 else "not significant"
print(f"   Top minus bottom: {top.mean() - bot.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}) -> {verdict}")

print("\n2. By verdict (M6)")
for k in ["Best Buy", "Fair", "Overpaid"]:
    g = s[s.verdict == k]
    if len(g) < 5:
        print(f"   {k:9s} n={len(g)} (too few to judge)"); continue
    lo, hi = boot_mean(g.on_off)
    print(f"   {k:9s} n={len(g):4d}  avg fee EUR {g.transfer_fee.mean() / 1e6:5.1f}M   realised on/off {g.on_off.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
    out.append({"test": "verdict", "group": k, "n": len(g), "predicted_points": g.points_mean.mean(),
                "realised_on_off": g.on_off.mean(), "ci_lo": lo, "ci_hi": hi})

from scipy.stats import spearmanr
rho, pv = spearmanr(s.points_mean, s.on_off)
rho_f, pv_f = spearmanr(s.transfer_fee, s.on_off)
print(f"\n3. Spearman(predicted points, realised on/off) = {rho:+.3f} (p = {pv:.3g})")
print(f"   Spearman(fee,              realised on/off) = {rho_f:+.3f} (p = {pv_f:.3g})   <- what the market price alone says")
out.append({"test": "spearman", "group": "predicted points", "n": len(s), "realised_on_off": rho})
out.append({"test": "spearman", "group": "fee", "n": len(s), "realised_on_off": rho_f})
pd.DataFrame(out).to_csv(config.REPORTS / "m7_validation.csv", index=False)
print("\nSaved reports/m7_validation.csv")
