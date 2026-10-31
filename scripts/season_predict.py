"""Season prediction: before a season starts, will a club finish in the TOP HALF of its league table?

This tests the whole FreeAgent chain (M2 prior -> M3 ratings -> M4 match model -> M5 season simulator) against
real final tables. It is also the first independent check of the season simulator.

FAIR-TEST RULES (fixed before running):
  * Everything uses only information from before 1 August of the season: ratings fitted on earlier matches,
    the squad from the club's last matches before 1 July plus the summer transfers up to 31 July, a match model
    trained on earlier seasons, market values dated before 1 August. The fixture list is known in advance.
  * NOTHING is tuned. The rule is fixed: predict "top half" for the clubs ranked in the top half by expected points
    (so every method predicts exactly half the league as top half, like the real table).
  * Baselines get the same information: (1) same half as last season (promoted clubs -> bottom half),
    (2) squad market value rank, (3) our own model WITHOUT the summer transfers, (4) a coin flip (50%).
  * Paired bootstrap over club-seasons gives a 95% CI for ours minus each baseline.

PROMOTED CLUBS (--promoted relegated, tested 30 Oct 2026 against the original rule, reported either way):
  our data has no second-division matches, so a newly promoted club has either no line-up at all (it then got
  the league-average strength 0) or a line-up from years ago. Rule: every club that was NOT in the league last
  season gets the average pre-season strength of the clubs it replaced (last season's relegated clubs).

Usage: python scripts/season_predict.py               # real data, test seasons 2019-2025
       python scripts/season_predict.py --promoted relegated   # same test with the promoted-club rule
       python scripts/season_predict.py --synthetic   # quick check on the fake world
Outputs (reports/): season_predictions.csv, season_summary.csv, season_by_league.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, match, simulate as M, success as S
from freeagent.impact import PlusMinus
from freeagent.prior import Prior

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--seasons", nargs="+", type=int, default=list(range(2019, 2026)))
p.add_argument("--sims", type=int, default=2000)
p.add_argument("--synthetic", action="store_true")
p.add_argument("--promoted", choices=["none", "relegated"], default="none")
args = p.parse_args()
t0 = time.time()
tick = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
rng = np.random.default_rng(config.SEED)

if args.synthetic:
    from freeagent.synthetic import make_world
    tables = data.standardise(make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5))
    comps = sorted(tables["games"].competition_id.unique())
    LEAGUES, LAM, HL = comps, 20.0, 730.0
    args.seasons, args.sims = [2019, 2020, 2021], 400
    d = data.prepare(tables, competitions=comps, first_season=2014)
    OUT = "season_synthetic_"
else:
    tables = data.load_tables(args.data)
    d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
    LEAGUES, LAM, HL = config.TOP_LEAGUES, 100.0, 1460.0
    OUT = "season_"
if args.promoted != "none":
    OUT += "promoted_"
G, ST, A, PL, TR = d["games"], d["stints"], d["appearances"], tables["players"], tables["transfers"]
VAL = tables.get("player_valuations", pd.DataFrame(columns=["player_id", "date", "market_value_in_eur"]))
lt = S.league_tables(tables)
tick(f"loaded {len(G):,} games")

cache = {}


def rated(cut):
    cut = pd.Timestamp(cut)
    if cut not in cache:
        pri = Prior(LAM, HL).fit(G, ST, A, PL, cut)
        pm = PlusMinus(lam=LAM, halflife_days=HL).fit(G, ST, cut, prior=pri.mu)
        cache[cut] = (pm, pri.mu)
    return cache[cut]


first = min(args.seasons)
st = match.honest_strength(G, ST, range(first - 4, max(args.seasons) + 1), lambda cut: rated(cut)[0].ratings)
GS = G.merge(st, on="game_id", how="left")
tick("cross-fitted strengths ready")


def final_table(season, league):
    g = G[(G.season == season) & (G.competition_id == league)].dropna(subset=["home_club_goals"])
    h = g[["home_club_id", "home_club_goals", "away_club_goals"]].set_axis(["club_id", "gf", "ga"], axis=1)
    a = g[["away_club_id", "away_club_goals", "home_club_goals"]].set_axis(["club_id", "gf", "ga"], axis=1)
    x = pd.concat([h, a])
    x["pts"] = np.select([x.gf > x.ga, x.gf == x.ga], [3, 1], 0)
    t = x.groupby("club_id").agg(points=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"), played=("pts", "size"))
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["points", "gd", "gf"], ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def arrival_shares(before_season):
    """Average first-season league minutes share of summer arrivals in earlier seasons (fee >= EUR 5M vs others)."""
    tr = TR.dropna(subset=["transfer_date"])
    tr = tr[tr.transfer_date.dt.month.isin([6, 7]) & (tr.transfer_date.dt.year < before_season)].copy()
    tr["season"] = tr.transfer_date.dt.year
    pcs = lt["pcs"][["player_id", "player_club_id", "season", "share"]]
    m = tr.merge(pcs, left_on=["player_id", "to_club_id", "season"], right_on=["player_id", "player_club_id", "season"])
    big = m.transfer_fee.fillna(0) >= config.MIN_FEE_EUR
    return float(m.share[big].mean()) if big.any() else 0.4, float(m.share[~big].mean()) if (~big).any() else 0.2


def squad_value(sq, cut):
    if len(VAL) == 0:
        return pd.Series(np.nan, index=sq.club_id.unique())
    v = VAL.dropna(subset=["date", "market_value_in_eur"])
    v = v[v.date < cut].sort_values("date").groupby("player_id").market_value_in_eur.last()
    return (sq.player_id.map(v).fillna(0) * sq.share).groupby(sq.club_id).sum()


def top_half_by_rank(score: pd.Series, n: int) -> pd.Series:
    """Predict top half for the n // 2 clubs with the highest score. Ties are broken by club id, never by the
    order of the real final table (that would leak the answer)."""
    s = score.sort_index()
    order = s.rank(ascending=False, method="first")
    return (order <= n // 2).astype(int).reindex(score.index)


rows = []
for Y in args.seasons:
    cut = pd.Timestamp(f"{Y}-08-01")
    pm, mu = rated(cut)
    model = match.PlayerPoisson().fit(GS, cut)
    share_big, share_other = arrival_shares(Y)
    base_sq = M.baseline_squad(G, ST, pm.ratings, PL, Y)
    # arrivals: one share for the whole window (we do not know who will start); use the fee-weighted mix later if needed
    new_sq = M.season_squad(G, ST, pm.ratings, PL, TR, Y, new_share=(share_big + share_other) / 2, fallback=mu)
    S_new, S_base = M.club_strength(new_sq), M.club_strength(base_sq)
    value = squad_value(new_sq, cut)
    for lg in LEAGUES:
        fx = G[(G.season == Y) & (G.competition_id == lg)]
        if fx.empty:
            continue
        real = final_table(Y, lg)
        n = len(real)
        prev = final_table(Y - 1, lg) if ((G.season == Y - 1) & (G.competition_id == lg)).any() else None
        S_use, S0_use = S_new, S_base
        if args.promoted == "relegated" and prev is not None:
            S_use, S0_use = (M.promoted_strength(x, real.index, prev.index) for x in (S_new, S_base))
        sim = M.simulate_table(model, fx, S_use, n_sims=args.sims, seed=config.SEED)
        sim0 = M.simulate_table(model, fx, S0_use, n_sims=args.sims, seed=config.SEED)
        clubs = real.index
        prev_top = pd.Series(0, index=clubs)
        prev_pts = pd.Series(np.nan, index=clubs)
        if prev is not None:
            m_prev = prev.reindex(clubs)
            prev_top = (m_prev["rank"] <= len(prev) // 2).fillna(False).astype(int)   # promoted -> bottom half
            prev_pts = m_prev.points * (real.played.max() / max(prev.played.max(), 1))
        out = pd.DataFrame({
            "season": Y, "league": lg, "club_id": clubs, "n_clubs": n,
            "actual_points": real.points.to_numpy(), "actual_rank": real["rank"].to_numpy(),
            "actual_top_half": (real["rank"] <= n // 2).astype(int).to_numpy(),
            "pred_points": sim.mean_points.reindex(clubs).to_numpy(),
            "p_top_half": sim.p_top_half.reindex(clubs).to_numpy(),
            "p_relegation": sim.p_relegation.reindex(clubs).to_numpy(),
            "actual_relegated": (real["rank"] > n - 3).astype(int).to_numpy(),
            "pred_top_half": top_half_by_rank(sim.mean_points.reindex(clubs), n).to_numpy(),
            "nosummer_top_half": top_half_by_rank(sim0.mean_points.reindex(clubs), n).to_numpy(),
            "value_top_half": top_half_by_rank(value.reindex(clubs).fillna(0), n).to_numpy(),
            "lastseason_top_half": prev_top.to_numpy(),
            "lastseason_points": prev_pts.to_numpy(),
            "promoted": (prev is not None) * (~clubs.isin(prev.index if prev is not None else [])).astype(int),
        })
        rows.append(out)
    tick(f"season {Y}: simulated {len(LEAGUES)} leagues (arrival share {share_big:.2f} / {share_other:.2f}, k = {model.coef[2]:.3f})")

res = pd.concat(rows, ignore_index=True)
if "clubs" in tables and "name" in tables["clubs"]:
    res["club_name"] = res.club_id.map(tables["clubs"].set_index("club_id").name)
res.to_csv(config.REPORTS / f"{OUT}predictions.csv", index=False)
y = res.actual_top_half.to_numpy()

methods = {"OURS: ratings + summer transfers + season simulation": "pred_top_half",
           "Ours without summer transfers": "nosummer_top_half",
           "Same half as last season": "lastseason_top_half",
           "Squad market value": "value_top_half"}
if len(VAL) == 0:
    methods.pop("Squad market value")          # the synthetic world has no market values
hits = {m: (res[c].to_numpy() == y).astype(float) for m, c in methods.items()}
hits["Coin flip (expected)"] = np.full(len(y), 0.5)


def ci(h, n=2000):
    b = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


OURS = list(methods)[0]
summ = []
for m, h in hits.items():
    lo, hi = ci(h)
    row = {"method": m, "accuracy": h.mean(), "ci_lo": lo, "ci_hi": hi, "n": len(h)}
    if m != OURS:
        diff = hits[OURS] - h
        dlo, dhi = ci(diff)
        row.update(ours_minus_this=diff.mean(), diff_lo=dlo, diff_hi=dhi,
                   verdict="OURS BETTER" if dlo > 0 else "ours worse" if dhi < 0 else "tied")
    summ.append(row)
summ = pd.DataFrame(summ).set_index("method")
summ.to_csv(config.REPORTS / f"{OUT}summary.csv")

print(f"\n=== Top half or bottom half, predicted before the season: {len(res)} club-seasons, "
      f"seasons {args.seasons[0]}-{args.seasons[-1]} ===")
print(summ.round(3).to_string())

# extra views
ok = res.lastseason_points.notna()
pts_mae = np.abs(res.pred_points - res.actual_points).mean()
base_mae = np.abs(res.lastseason_points[ok] - res.actual_points[ok]).mean()
ours_mae_same = np.abs(res.pred_points[ok] - res.actual_points[ok]).mean()
rho = np.mean([spearmanr(g.pred_points, g.actual_points)[0] for _, g in res.groupby(["season", "league"])])
rho_prev = np.mean([spearmanr(g.lastseason_points, g.actual_points, nan_policy="omit")[0]
                    for _, g in res[ok].groupby(["season", "league"])])
brier = np.mean((res.p_top_half - y) ** 2)
rel_brier = np.mean((res.p_relegation - res.actual_relegated) ** 2)
calib = res.assign(bin=pd.cut(res.p_top_half, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)).groupby(
    "bin", observed=True).agg(n=("actual_top_half", "size"), predicted=("p_top_half", "mean"),
                              actual=("actual_top_half", "mean"))
print(f"\nFinal points: mean absolute error {pts_mae:.1f} points (ours). On clubs that were in the league last season: "
      f"ours {ours_mae_same:.1f} vs last season's points {base_mae:.1f}")
print(f"Rank correlation with the final table (Spearman, average per league-season): ours {rho:.3f}, "
      f"last season's points {rho_prev:.3f} (non-promoted clubs only)")
print(f"Brier score of P(top half): {brier:.3f} (coin flip 0.250);  Brier of P(relegation): {rel_brier:.3f}")
print("\nCalibration: when we say X% chance of top half, how often does it happen?")
print(calib.round(3).to_string())
bl = res.groupby("league").apply(lambda g: pd.Series({
    "n": len(g), "ours": (g.pred_top_half == g.actual_top_half).mean(),
    "same_as_last_season": (g.lastseason_top_half == g.actual_top_half).mean(),
    "market_value": (g.value_top_half == g.actual_top_half).mean()}), include_groups=False)
bs = res.groupby("season").apply(lambda g: pd.Series({
    "n": len(g), "ours": (g.pred_top_half == g.actual_top_half).mean(),
    "same_as_last_season": (g.lastseason_top_half == g.actual_top_half).mean(),
    "market_value": (g.value_top_half == g.actual_top_half).mean()}), include_groups=False)
bl.to_csv(config.REPORTS / f"{OUT}by_league.csv")
print("\nBy league:")
print(bl.round(3).to_string())
print("\nBy season:")
print(bs.round(3).to_string())
pr = res[res.promoted == 1]
if len(pr):
    print(f"\nPromoted clubs ({len(pr)}): ours {(pr.pred_top_half == pr.actual_top_half).mean():.3f}, "
          f"'same half as last season' {(pr.lastseason_top_half == pr.actual_top_half).mean():.3f}")
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv")
