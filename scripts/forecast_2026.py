"""FreeAgent 2026/27: the LIVE forecast (the product). Past seasons are only the evidence that the method works.

What this makes, using only data up to the July 2026 snapshot (last match 24 May 2026, transfers to early July):
  1. Season forecast 2026/27 for the top five leagues: every club's expected points, 80% range, title, top-4,
     top-half and relegation chances (the season is played 5,000 times).
  2. Summer 2026 signings (EUR 5M+): chance he becomes a regular (M8), league points he adds (M5) and the
     verdict on the fee (M6). These outcomes are unknown today, so these are genuine predictions.
  3. Transfer Lab tables: each club's squad and points curve, plus rated players, so the app can answer
     "what would player X add to club Y for fee Z" without running any model.

NOTHING is re-tuned for 2026: every setting is the one tested walk-forward on 2019-2025 (season_predict.py,
m5_points_per_signing.py, m8). The models are simply trained on all the data we have, as they would be on
transfer day. 2026/27 line-ups: the official clubs of each league (relegation and promotion of May 2026,
including play-offs), checked against Wikipedia's 2026-27 league pages; the fixture list is a double round
robin (the order of matches does not change expected points).

Usage: python scripts/forecast_2026.py             (real data, about 10 minutes)
       python scripts/forecast_2026.py --synthetic (quick check on the fake world)
Outputs (reports/): forecast_2026_table.csv, forecast_2026_signings.csv, forecast_2026_moves.csv,
                    lab_squads.csv, lab_curves.csv, lab_players.csv, forecast_2026_meta.csv
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

# 2026/27 line-ups = 2025/26 clubs - relegated + promoted (club ids from the data / Transfermarkt)
RELEGATED_2025 = {"GB1": [379, 1132, 543],          # West Ham, Burnley, Wolves
                  "ES1": [237, 12321, 2497],        # Mallorca, Girona, Real Oviedo
                  "IT1": [2239, 276, 4172],         # Cremonese, Hellas Verona, Pisa
                  "L1": [2036, 35, 82],             # Heidenheim, St. Pauli, Wolfsburg (lost the play-off)
                  "FR1": [347, 995]}                # Metz, Nantes
PROMOTED_2026 = {"GB1": [990, 677, 3008],           # Coventry City, Ipswich Town, Hull City
                 "ES1": [630, 897, 1084],           # Racing Santander, Deportivo La Coruna, Malaga
                 "IT1": [607, 8970, 2919],          # Venezia, Frosinone, Monza
                 "L1": [33, 64, 127],               # Schalke 04, SV Elversberg, SC Paderborn (won the play-off)
                 "FR1": [1095, 1164]}               # Troyes, Le Mans
LEAGUE_SIZE = {"GB1": 20, "ES1": 20, "IT1": 20, "L1": 18, "FR1": 18}

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--sims", type=int, default=5000)
p.add_argument("--draws", type=int, default=4000)
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()
t0 = time.time()
tick = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
rng = np.random.default_rng(config.SEED)

if args.synthetic:
    from freeagent.synthetic import make_world
    tables = data.standardise(make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5))
    # like the real data: the season to forecast (2021) has transfers but no matches yet
    gid = tables["games"].game_id[tables["games"].season < 2021]
    tables["games"] = tables["games"][tables["games"].game_id.isin(gid)]
    for k in ("appearances", "game_lineups"):
        if k in tables:
            tables[k] = tables[k][tables[k].game_id.isin(gid)]
    comps = sorted(tables["games"].competition_id.unique())
    d = data.prepare(tables, competitions=comps, first_season=2014)
    LEAGUES, LAM, HL, Y = comps, 20.0, 730.0, 2021
    args.sims, args.draws = 400, 500
    OUT = "forecast_synthetic_"
else:
    tables = data.load_tables(args.data)
    d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
    LEAGUES, LAM, HL, Y = config.TOP_LEAGUES, 100.0, 1460.0, 2026
    OUT = "forecast_2026_"
G, ST, A, PL, TR = d["games"], d["stints"], d["appearances"], tables["players"], tables["transfers"]
VAL = tables.get("player_valuations", pd.DataFrame(columns=["player_id", "date", "market_value_in_eur"]))
CUT = pd.Timestamp(f"{Y}-08-01")
tick(f"loaded {len(G):,} games, last match {G.date.max():%d %b %Y}")

# ---------------------------------------------------------------- club names (clubs.csv misses some promoted clubs)
names = pd.Series(dtype=object)
if "name" in tables["clubs"]:
    names = tables["clubs"].set_index("club_id").name
if "to_club_name" in TR:
    tn = pd.concat([TR[["to_club_id", "to_club_name"]].set_axis(["id", "n"], axis=1),
                    TR[["from_club_id", "from_club_name"]].set_axis(["id", "n"], axis=1)]).dropna()
    names = names.combine_first(tn.drop_duplicates("id").set_index("id").n)
name = lambda c: str(names.get(c, f"Club {c}"))

# ---------------------------------------------------------------- 2026/27 line-ups and fixtures
last = Y - 1
clubs_now, clubs_last = {}, {}
for lg in LEAGUES:
    prev = sorted(set(G[(G.season == last) & (G.competition_id == lg)].home_club_id))
    clubs_last[lg] = prev
    if args.synthetic:
        clubs_now[lg] = prev
    else:
        now = sorted((set(prev) - set(RELEGATED_2025[lg])) | set(PROMOTED_2026[lg]))
        assert len(now) == LEAGUE_SIZE[lg], (lg, len(now))
        clubs_now[lg] = now
fixtures = {lg: pd.DataFrame([(h, a) for h in cl for a in cl if h != a], columns=["home_club_id", "away_club_id"])
            .assign(competition_id=lg, season=Y) for lg, cl in clubs_now.items()}
league_of = {c: lg for lg, cl in clubs_now.items() for c in cl}
promoted = {c for lg in LEAGUES for c in clubs_now[lg] if c not in clubs_last[lg]}
tick(f"{sum(map(len, clubs_now.values()))} clubs in {len(LEAGUES)} leagues, {len(promoted)} promoted")

# ---------------------------------------------------------------- frozen models, trained on everything before the season
cache = {}


def rated(cut):
    cut = pd.Timestamp(cut)
    if cut not in cache:
        pri = Prior(LAM, HL).fit(G, ST, A, PL, cut)
        pm = PlusMinus(lam=LAM, halflife_days=HL).fit(G, ST, cut, prior=pri.mu)
        cache[cut] = (pm, pri.mu)
    return cache[cut]


st_ = match.honest_strength(G, ST, range(Y - 6, Y), lambda c: rated(c)[0].ratings)
GS = G.merge(st_, on="game_id", how="left")
model = match.PlayerPoisson().fit(GS, CUT)
pm, mu = rated(CUT)
tick(f"ratings for {len(pm.ratings):,} players; match model k = {model.coef[2]:.3f}")

lt = S.league_tables(tables)
share_new = M.arrival_share(TR, lt["pcs"], Y)
sq = M.season_squad(G, ST, pm.ratings, PL, TR, Y, new_share=share_new, fallback=mu)
S_all = M.club_strength(sq)
S_use = S_all.copy()
for lg in LEAGUES:                                   # promoted-club rule (tested in season_predict.py --promoted)
    S_use = M.promoted_strength(S_use, clubs_now[lg], clubs_last[lg])

# ---------------------------------------------------------------- 1. season forecast
rows = []
for lg in LEAGUES:
    sim = M.simulate_table(model, fixtures[lg], S_use, n_sims=args.sims, seed=config.SEED)
    sim = sim.reindex(clubs_now[lg])
    sim = sim.assign(league=lg, club_name=[name(c) for c in sim.index], strength=S_use.reindex(sim.index).to_numpy(),
                     promoted=[int(c in promoted) for c in sim.index])
    rows.append(sim.reset_index())
table = pd.concat(rows, ignore_index=True).sort_values(["league", "mean_points"], ascending=[True, False])
table["pred_rank"] = table.groupby("league").mean_points.rank(ascending=False, method="first").astype(int)
tick("season forecast done")

# summer moves per club (what changed since May), for the forecast page
tr = TR.dropna(subset=["transfer_date"])
win = tr[(tr.transfer_date >= pd.Timestamp(f"{Y}-06-01")) & (tr.transfer_date < CUT)]
moves = []
for c in table.club_id:
    for kind, part in (("in", win[win.to_club_id == c]), ("out", win[win.from_club_id == c])):
        part = part[part.transfer_fee.fillna(0) > 0].sort_values("transfer_fee", ascending=False)
        for r in part.head(4).itertuples():
            moves.append({"club_id": c, "kind": kind, "player_id": r.player_id,
                          "player_name": getattr(r, "player_name", ""), "fee_m": r.transfer_fee / 1e6,
                          "other_club": getattr(r, "from_club_name" if kind == "in" else "to_club_name", "")})
moves = pd.DataFrame(moves, columns=["club_id", "kind", "player_id", "player_name", "fee_m", "other_club"])

# ---------------------------------------------------------------- 2. summer signings of this window
s = tr[(tr.transfer_fee >= config.MIN_FEE_EUR) & tr.transfer_date.dt.month.isin(S.SUMMER_MONTHS)
       & (tr.transfer_date.dt.year == Y) & tr.to_club_id.isin(league_of)].copy()
s["season"] = Y
s["to_league"] = s.to_club_id.map(league_of)
s["share"], s["regular"] = np.nan, np.nan
for c in ["player_name", "from_club_name", "to_club_name"]:
    if c not in s:
        s[c] = np.nan
s = s.drop_duplicates(["player_id", "season", "to_club_id"])
keep = ["player_id", "player_name", "transfer_date", "season", "from_club_id", "to_club_id",
        "from_club_name", "to_club_name", "to_league", "transfer_fee", "share", "regular"]
hist = S.build_signings(tables, lt, last_season=Y - 1)
r8 = S.fit_ratings(G, ST, range(2015, Y + 1), lam=LAM, halflife_days=HL)
feat = S.build_features(pd.concat([hist, s[keep]], ignore_index=True), tables, lt, r8)
FEATS = [f for f in S.NUMERIC + S.CATEGORICAL if feat[f].notna().any()]   # synthetic world: no market values
train = feat[(feat.season >= 2015) & (feat.season < Y)]
now = feat[feat.season == Y].copy()
m8 = S.Ensemble(FEATS, max_depth=3, learning_rate=0.03, max_iter=150).fit(train[FEATS], train.regular.astype(int))
now["p"] = m8.predict_proba(now[FEATS])[:, 1] if len(now) else []
tick(f"M8 trained on {len(train):,} past signings, predicting {len(now)} signings of {Y}")

if len(VAL):          # same two market-parameter estimates as M5 (club-seasons 2015-2025)
    LOW = V.points_per_million(G, ST, PL, VAL, lt["club"], range(2015, Y), within_club=True)
    HIGH = V.points_per_million(G, ST, PL, VAL, lt["club"], range(2015, Y), within_club=False)
else:                 # the synthetic world has no market values: use the real estimates
    LOW, HIGH = {"b": 0.01268}, {"b": 0.067}
sd0 = float(np.sqrt(pm.sigma2 / pm.lam))
pool_reg = train.share[train.regular == 1].to_numpy()
pool_other = train.share[train.regular == 0].to_numpy()
pos = PL.set_index("player_id").position
out = []
for r in now.itertuples():
    pid, c = r.player_id, int(r.to_club_id)
    base = {"player_id": pid, "player_name": r.player_name, "season": Y, "to_club_id": c,
            "to_club_name": r.to_club_name if isinstance(r.to_club_name, str) else name(c),
            "from_club_name": r.from_club_name, "to_league": r.to_league, "transfer_fee": r.transfer_fee,
            "transfer_date": r.transfer_date, "p_regular": r.p, "regular": np.nan, "actual_share": np.nan,
            "promoted_club": int(c in promoted)}
    known = pid in pm.ratings.index
    if c in promoted or c not in S_all.index:
        out.append({**base, "known_player": int(known), "verdict": "Not rated", "note": "newly promoted club: no recent line-up in our data"})
        continue
    rating = float(pm.ratings[pid]) if known else float(mu.get(pid, 0.0))
    sd = float(pm.sd([pid]).iloc[0]) if known else sd0
    dl, gain, r_inc = M.signing_points(model, fixtures[r.to_league], sq, S_use, c, pid, pos.get(pid, "Unknown"),
                                       rating, sd, r.p, pool_reg, pool_other, args.draws, rng)
    out.append({**base, "rating": rating, "rating_sd": sd, "known_player": int(known), "r_displaced": r_inc,
                "delta_mean": dl.mean(), "points_mean": gain.mean(), "points_sd": gain.std(),
                "points_p05": np.percentile(gain, 5), "points_p95": np.percentile(gain, 95),
                **{f"gq_{i:02d}": v for i, v in enumerate(np.percentile(gain, V.QUANTILE_LEVELS))},
                "price_b_per_100m": 100 * LOW["b"], "price_high_per_100m": 100 * HIGH["b"],
                **V.verdict_probs(gain, r.transfer_fee, LOW["b"]),
                **{k + "_high": v for k, v in V.verdict_probs(gain, r.transfer_fee, HIGH["b"]).items()}})
sig = pd.DataFrame(out)
tick(f"{len(sig)} signings scored")

# ---------------------------------------------------------------- 3. Transfer Lab tables
lab_sq = sq[sq.club_id.isin(league_of)].copy()
lab_sq = lab_sq.assign(league=lab_sq.club_id.map(league_of),
                       player_name=lab_sq.player_id.map(PL.set_index("player_id").get("name", pd.Series(dtype=object))))
grid = np.round(np.linspace(-0.4, 0.4, 41), 4)
curves = []
for lg in LEAGUES:
    for c in clubs_now[lg]:
        pts = M.points_curve(model, fixtures[lg], S_use, c, grid)
        curves += [{"club_id": c, "delta": g_, "points": v} for g_, v in zip(grid, pts)]
curves = pd.DataFrame(curves)
def sd_diag(pm, players):
    """Posterior sd of each player's rating: one sparse solve per player, diagonal only (pm.sd builds the whole
    covariance block, far too big for thousands of players)."""
    S0 = len(pm._seasons)
    out = []
    for q in players:
        e = np.zeros(S0 + len(pm._players)); e[S0 + pm._pcol[q]] = 1.0
        out.append(np.sqrt(pm.sigma2 * pm._lu.solve(e)[S0 + pm._pcol[q]]))
    return np.array(out)


active = set(A[A.date >= CUT - pd.Timedelta(days=550)].player_id)          # played since early 2025
mins = pm.minutes.reindex(pm.ratings.index)
cand = pd.DataFrame({"player_id": pm.ratings.index, "rating": pm.ratings.to_numpy(), "minutes": mins.to_numpy()})
cand = cand[(cand.minutes >= 900) & cand.player_id.isin(active) & cand.player_id.isin(pm._pcol.index)]
cand["rating_sd"] = sd_diag(pm, cand.player_id)
pinfo = PL.set_index("player_id")
cand["player_name"] = cand.player_id.map(pinfo.get("name", pd.Series(dtype=object)))
cand["position"] = cand.player_id.map(pinfo.position).fillna("Unknown")
cand["age"] = ((CUT - pd.to_datetime(cand.player_id.map(pinfo.date_of_birth))).dt.days / 365.25).round(1)
last_club = sq.drop_duplicates("player_id", keep="last").set_index("player_id").club_id
cand["club_id"] = cand.player_id.map(last_club)
cand["club_name"] = [name(c) if pd.notna(c) else "" for c in cand.club_id]
if len(VAL):
    v = VAL.dropna(subset=["date", "market_value_in_eur"])
    cand["value_m"] = cand.player_id.map(v[v.date < CUT].sort_values("date").groupby("player_id").market_value_in_eur.last()) / 1e6
cand = cand.dropna(subset=["player_name"]) if cand.player_name.notna().any() else cand
cand = cand.sort_values("rating", ascending=False)

# ---------------------------------------------------------------- save
config.REPORTS.mkdir(exist_ok=True)
table.to_csv(config.REPORTS / f"{OUT}table.csv", index=False)
sig.to_csv(config.REPORTS / f"{OUT}signings.csv", index=False)
moves.to_csv(config.REPORTS / f"{OUT}moves.csv", index=False)
lab = "lab_" if not args.synthetic else "lab_synthetic_"
lab_sq.to_csv(config.REPORTS / f"{lab}squads.csv", index=False)
curves.to_csv(config.REPORTS / f"{lab}curves.csv", index=False)
cand.to_csv(config.REPORTS / f"{lab}players.csv", index=False)
pd.Series({"season": Y, "last_match": f"{G.date.max():%Y-%m-%d}", "last_transfer": f"{win.transfer_date.max():%Y-%m-%d}"
           if len(win) else "", "sims": args.sims, "price_low": 100 * LOW["b"], "price_high": 100 * HIGH["b"],
           "arrival_share": share_new, "k": model.coef[2]}).to_csv(config.REPORTS / f"{OUT}meta.csv", header=["value"])

print(f"\n=== {Y}/{str(Y + 1)[-2:]} forecast (data to {G.date.max():%d %b %Y}) ===")
for lg in LEAGUES:
    t = table[table.league == lg]
    print(f"\n{lg}: " + " | ".join(f"{r.club_name} {r.mean_points:.0f} ({100 * r.p_title:.0f}%)" for r in t.head(4).itertuples()))
    print("   relegation: " + ", ".join(f"{r.club_name} {100 * r.p_relegation:.0f}%" for r in
                                       t.sort_values("p_relegation", ascending=False).head(3).itertuples()))
if len(sig):
    print(f"\nSummer {Y} signings: {len(sig)}  verdicts: {sig.verdict.value_counts().to_dict()}")
    show = sig.sort_values("transfer_fee", ascending=False).head(12)
    cols = [c for c in ["player_name", "to_club_name", "transfer_fee", "p_regular", "points_mean", "claimed_points",
                        "verdict", "confidence"] if c in show]
    print(show[cols].round(2).to_string(index=False))
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv and reports/{lab}*.csv")
