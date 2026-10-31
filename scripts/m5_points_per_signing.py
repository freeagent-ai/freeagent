"""M5 on real data: for every big signing in the test seasons, predict the league points he adds.

Uses only information available before 1 August of the signing year:
  ratings   : M3 Bayesian plus-minus with the M2 ML prior, fitted on matches before 1 Aug
  match model: M4 player Poisson, trained on cross-fitted strengths from earlier seasons
  p_regular : M8 classifier, trained on earlier seasons only (walk-forward)
Output: reports/m5_signing_points.csv (one row per signing) = the input of M6.

Method v2 (default, fixed 30 Oct 2026 after a bias check, see below):
  squad      : the summer-aware pre-season squad (players who left by 31 July removed, arrivals added),
               the same squad the season forecast uses
  displaced  : a signing takes his minutes from the WEAKEST players first (same position, then the rest)
Method v1 (--method v1, the original): squad from the last 5 matches before 1 July (ignores the summer) and
  he displaces the minutes-weighted AVERAGE player in his position. v1 called 46% of the signings by the
  strongest quarter of clubs "Overpaid" against 2% for the weakest quarter, because an elite club's average
  incumbent is a star while the player who really loses his place is the weakest one.

Usage: python scripts/m5_points_per_signing.py [--seasons 2019 2020 ...] [--draws 4000] [--method v1|v2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, match, simulate as M, success as S, value as V
from freeagent.impact import PlusMinus
from freeagent.prior import Prior

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--seasons", nargs="+", type=int, default=list(range(2019, 2026)))
p.add_argument("--draws", type=int, default=4000)
p.add_argument("--lam", type=float, default=100.0)
p.add_argument("--halflife", type=float, default=1460.0)
p.add_argument("--method", choices=["v1", "v2"], default="v2")
args = p.parse_args()
t0 = time.time()
LAM, HL = args.lam, args.halflife
tick = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

tables = data.load_tables(args.data)
d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
G, ST, A, PL = d["games"], d["stints"], d["appearances"], tables["players"]
tick(f"loaded {len(G):,} games")

# ---- M8: out-of-sample probability that each signing becomes a regular
lt = S.league_tables(tables)
sig = S.build_signings(tables, lt)
r8 = S.fit_ratings(G, ST, range(2015, max(args.seasons) + 1), lam=LAM, halflife_days=HL)
df = S.build_features(sig, tables, lt, r8)
df = df[df.season >= 2015].reset_index(drop=True)
FEATS = S.NUMERIC + S.CATEGORICAL
pr8 = S.walk_forward(df, lambda: S.Ensemble(FEATS, max_depth=3, learning_rate=0.03, max_iter=150), FEATS, args.seasons)
tick(f"M8 predictions for {len(pr8)} signings")

# ---- M2 + M3: ratings with ML prior, cached per cutoff
cache = {}


def rated(cut):
    cut = pd.Timestamp(cut)
    if cut not in cache:
        pri = Prior(LAM, HL).fit(G, ST, A, PL, cut)
        pm = PlusMinus(lam=LAM, halflife_days=HL).fit(G, ST, cut, prior=pri.mu)
        cache[cut] = (pm, pri.mu)
    return cache[cut]


# ---- M4: match model trained on cross-fitted strengths
first = min(args.seasons)
st = match.honest_strength(G, ST, range(first - 4, max(args.seasons) + 1), lambda cut: rated(cut)[0].ratings)
GS = G.merge(st, on="game_id", how="left")
tick("cross-fitted strengths ready")

# M6: the price of a league point is NOT well identified (see README). We carry two market-parameter
# estimates, both from all club-seasons 2015-2025 (a market parameter, not a prediction target):
#   LOW  = club fixed effects (compare a club only with itself over time)      -> conservative
#   HIGH = between clubs (rich vs poor clubs; inflated by 'rich clubs win more') -> upper bound
LOW = V.points_per_million(G, ST, PL, tables["player_valuations"], lt["club"], range(2015, 2026), within_club=True)
HIGH = V.points_per_million(G, ST, PL, tables["player_valuations"], lt["club"], range(2015, 2026), within_club=False)
tick(f"price of a point: LOW {100 * LOW['b']:.2f} (se {100 * LOW['se']:.2f}), HIGH {100 * HIGH['b']:.2f} points per EUR 100M")

rng = np.random.default_rng(config.SEED)
rows = []
for Y in args.seasons:
    cut = pd.Timestamp(f"{Y}-08-01")
    pm, mu = rated(cut)
    model = match.PlayerPoisson().fit(GS, cut)
    if args.method == "v2":
        sq = M.season_squad(G, ST, pm.ratings, PL, tables["transfers"], Y,
                            new_share=M.arrival_share(tables["transfers"], lt["pcs"], Y), fallback=mu)
    else:
        sq = M.baseline_squad(G, ST, pm.ratings, PL, Y)
    Sc = M.club_strength(sq)
    sd0 = float(np.sqrt(pm.sigma2 / pm.lam))                  # prior sd for players we know nothing about
    train = df[df.season < Y]
    pool_reg = train.share[train.regular == 1].to_numpy()
    pool_other = train.share[train.regular == 0].to_numpy()
    sg = pr8[pr8.season == Y]
    pos = PL.set_index("player_id").position
    for r in sg.itertuples():
        pid = r.player_id
        fx = G[(G.season == Y) & (G.competition_id == r.to_league)]
        if r.to_club_id not in Sc.index or fx.empty:
            continue
        known = pid in pm.ratings.index
        rating = float(pm.ratings[pid]) if known else float(mu.get(pid, 0.0))
        sd = float(pm.sd([pid]).iloc[0]) if known else sd0
        if args.method == "v2":
            dl, gain, r_inc = M.signing_points(model, fx, sq, Sc, int(r.to_club_id), pid, pos.get(pid, "Unknown"),
                                               rating, sd, r.p, pool_reg, pool_other, args.draws, rng)
        else:
            r_inc = M.displaced_rating(sq, r.to_club_id, pos.get(pid, "Unknown"), exclude=pid)
            dl = M.delta_draws(rating, sd, r.p, r_inc, pool_reg, pool_other, args.draws, rng)
            gain = M.points_gain(model, fx, Sc, int(r.to_club_id), dl)
        rows.append({"player_id": pid, "player_name": r.player_name, "season": Y, "to_club_id": r.to_club_id,
                     "to_club_name": r.to_club_name, "to_league": r.to_league, "transfer_fee": r.transfer_fee,
                     "p_regular": r.p, "regular": r.regular, "actual_share": r.share,
                     "rating": rating, "rating_sd": sd, "known_player": int(known), "r_displaced": r_inc,
                     "delta_mean": dl.mean(), "points_mean": gain.mean(), "points_sd": gain.std(),
                     "points_p05": np.percentile(gain, 5), "points_p95": np.percentile(gain, 95),
                     **{f"gq_{i:02d}": v for i, v in enumerate(np.percentile(gain, V.QUANTILE_LEVELS))},
                     "price_b_per_100m": 100 * LOW["b"], "price_high_per_100m": 100 * HIGH["b"],
                     **V.verdict_probs(gain, r.transfer_fee, LOW["b"]),
                     **{k + "_high": v for k, v in V.verdict_probs(gain, r.transfer_fee, HIGH["b"]).items()}})
    tick(f"season {Y}: {len(sg)} signings, k = {model.coef[2]:.3f}")

out = pd.DataFrame(rows)
config.REPORTS.mkdir(exist_ok=True)
name = "m5_signing_points.csv" if args.method == "v2" else "m5_signing_points_v1.csv"
out.to_csv(config.REPORTS / name, index=False)
print(f"\n{len(out)} signings with predicted points ({args.method}). Saved reports/{name}")
print(out[["transfer_fee", "p_regular", "rating", "rating_sd", "points_mean", "points_sd"]].describe().round(3).to_string())
print(f"\nKnown players (had a rating): {out.known_player.mean():.1%}")
for tag, sfx in (("LOW price (club fixed effects)", ""), ("HIGH price (between clubs)", "_high")):
    print(f"\nVerdicts (M6) at the {tag}:")
    print(out["verdict" + sfx].value_counts().to_string())
    print(out.groupby("verdict" + sfx)[["transfer_fee", "points_mean", "claimed_points" + sfx, "confidence" + sfx]].mean().round(2).to_string())
print("\nBiggest predicted gains:")
print(out.sort_values("points_mean", ascending=False).head(8)[
    ["player_name", "to_club_name", "season", "transfer_fee", "points_mean", "points_p05", "points_p95"]].round(2).to_string(index=False))
print("\nBiggest predicted losses:")
print(out.sort_values("points_mean").head(5)[
    ["player_name", "to_club_name", "season", "transfer_fee", "points_mean", "points_p05", "points_p95"]].round(2).to_string(index=False))
print(f"\nTotal time {time.time() - t0:.0f}s")
