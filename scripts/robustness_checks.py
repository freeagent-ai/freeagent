"""Robustness checks: do our conclusions survive reasonable changes to our own choices?

Parts (run any subset with --parts):
  threshold    M8 label: "regular" = at least 30/40/50/60/70% of the club's league minutes. Headline accuracy depends on the
               base rate, so we report the LIFT over the majority-class baseline for every threshold.
  position     M8 by position: accuracy/AUC of the frozen model per position, and per-position models vs one pooled model.
  halflife_m8  Rating half-life (730 / 1460 / 2920 / 10000 days): refit ratings, re-run the frozen M8 model.
  halflife_rq1 Same half-lives for the match model (RQ1 pooled log loss). SLOW (re-fits ratings many times).
This is a sensitivity analysis: nothing here is used to re-tune the headline model.

Usage: python scripts/robustness_checks.py --parts threshold position halflife_m8
       python scripts/robustness_checks.py --parts halflife_rq1
       python scripts/robustness_checks.py --synthetic        (quick check on the fake world)
Outputs (reports/): robustness_threshold.csv, robustness_position.csv, robustness_halflife_m8.csv, robustness_halflife_rq1.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, match, success as S
from freeagent.impact import PlusMinus

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--parts", nargs="+", default=["threshold", "position", "halflife_m8"],
               choices=["threshold", "position", "halflife_m8", "halflife_rq1"])
p.add_argument("--test", nargs="+", type=int, default=list(range(2019, 2026)))
p.add_argument("--halflives", nargs="+", type=float, default=[730, 1460, 2920, 10000])
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()
t0 = time.time()
rng = np.random.default_rng(config.SEED)
OUT = "robustness_synthetic_" if args.synthetic else "robustness_"

if args.synthetic:
    from freeagent.synthetic import make_world
    tables = data.standardise(make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5))
    args.test, args.halflives = [2019, 2020, 2021], [365, 730]
    comps = sorted(tables["games"].competition_id.unique())
    d = data.prepare(tables, competitions=comps, first_season=2014)
    rkw = dict(lam=20)
    HL0 = 730.0
else:
    tables = data.load_tables(args.data)
    comps = config.RATING_COMPETITIONS
    d = data.prepare(tables, competitions=comps)
    rkw = dict(lam=100.0)
    HL0 = 1460.0
lt = S.league_tables(tables)
sig = S.build_signings(tables, lt)
usable = lambda df_, fs: [f for f in fs if df_[f].notna().any()]


def make_df(hl):
    r = S.fit_ratings(d["games"], d["stints"], range(2015, max(args.test) + 1), halflife_days=hl, **rkw)
    x = S.build_features(sig, tables, lt, r)
    return x[x.season >= 2015].reset_index(drop=True)


def ours(feats):
    return lambda: S.Ensemble(feats, max_depth=3, max_iter=150, learning_rate=0.03)


def summarise(pr):
    pp, y = pr.p.to_numpy(), pr.regular.to_numpy()
    sc = S.scores(pp, y)
    m = np.abs(pp - 0.5) >= S.CONFIDENCE_MARGIN
    hit = ((pp >= 0.5).astype(int) == y)
    return {**sc, "confident_accuracy": hit[m].mean() if m.any() else np.nan, "coverage": m.mean()}


def boot_ci(h, n=1000):
    b = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


df = make_df(HL0)
BASE = usable(df, S.NUMERIC + S.CATEGORICAL)
print(f"{len(df)} signings, parts = {args.parts}  [{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ 1. label threshold
if "threshold" in args.parts:
    rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        x = df.assign(regular=(df.share >= thr).astype(int))
        pr = S.walk_forward(x, ours(BASE), BASE, args.test)
        mj = S.walk_forward(x, lambda: S.Majority(), BASE, args.test)
        a_h = ((pr.p >= 0.5).astype(int) == pr.regular).to_numpy().astype(float)
        a_m = ((mj.p >= 0.5).astype(int) == mj.regular).to_numpy().astype(float)
        diff = a_h - a_m
        lo, hi = np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)], [2.5, 97.5])
        s_ = summarise(pr)
        rows.append({"threshold": thr, "test_base_rate": pr.regular.mean(), "majority_accuracy": a_m.mean(),
                     "our_accuracy": a_h.mean(), "lift_vs_majority": diff.mean(), "lift_lo": lo, "lift_hi": hi,
                     "auc": s_["auc"], "log_loss": s_["log_loss"], "confident_accuracy": s_["confident_accuracy"],
                     "coverage": s_["coverage"], "n": len(pr)})
    th = pd.DataFrame(rows)
    th.to_csv(config.REPORTS / f"{OUT}threshold.csv", index=False)
    print("\n=== 1. 'Regular' threshold (frozen model, walk-forward; the lift over the majority class is what to compare) ===")
    print(th.round(3).to_string(index=False))

# ------------------------------------------------------------------ 2. by position
if "position" in args.parts:
    pooled = S.walk_forward(df, ours(BASE), BASE, args.test)
    pooled["pos"] = pooled.position.astype(str)
    feats_np = [f for f in BASE if f != "position"]
    MIN_GROUP = 30 if args.synthetic else 150
    parts_, rows = [], []
    for g_, dg in df.assign(pos=df.position.astype(str)).groupby("pos"):
        if len(dg) < MIN_GROUP:
            continue
        out = []
        for y_ in args.test:
            tr, te = dg[dg.season < y_], dg[dg.season == y_]
            if len(te) == 0:
                continue
            if len(tr) >= MIN_GROUP // 2 and tr.regular.nunique() == 2:
                m = ours(feats_np)().fit(tr[feats_np], tr.regular)
                out.append(te.assign(p_pos=m.predict_proba(te[feats_np])[:, 1], used_pos_model=1))
            else:
                out.append(te.assign(p_pos=pooled.p.reindex(te.index).to_numpy(), used_pos_model=0))
        parts_.append(pd.concat(out))
    pos = pd.concat(parts_)
    pos = pos.join(pooled[["p"]].rename(columns={"p": "p_pooled"}))
    for g_, q in pos.groupby("pos"):
        y = q.regular.to_numpy()
        ha_p, ha_s = ((q.p_pooled >= 0.5) == y), ((q.p_pos >= 0.5) == y)
        rows.append({"position": g_, "n": len(q), "base_rate": y.mean(),
                     "majority_accuracy": max(y.mean(), 1 - y.mean()),
                     "pooled_model_accuracy": ha_p.mean(), "position_model_accuracy": ha_s.mean(),
                     "pooled_auc": roc_auc_score(y, q.p_pooled) if len(np.unique(y)) > 1 else np.nan,
                     "rating_alone_auc": (roc_auc_score(y[q.rating.notna()], q.rating.dropna())
                                          if q.rating.notna().sum() > 30 and len(np.unique(y[q.rating.notna()])) > 1 else np.nan)})
    po = pd.DataFrame(rows)
    po.to_csv(config.REPORTS / f"{OUT}position.csv", index=False)
    print("\n=== 2. By position (test seasons; pooled model vs separate model per position) ===")
    print(po.round(3).to_string(index=False))
    y = pos.regular.to_numpy()
    diff = ((pos.p_pos >= 0.5) == y).to_numpy().astype(float) - ((pos.p_pooled >= 0.5) == y).to_numpy().astype(float)
    lo, hi = np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)], [2.5, 97.5])
    print(f"All positions together (n={len(pos)}): per-position models minus pooled model = {diff.mean():+.3f} "
          f"(95% CI {lo:+.3f} to {hi:+.3f})")

# ------------------------------------------------------------------ 3. rating half-life, M8
if "halflife_m8" in args.parts:
    rows = []
    for hl in args.halflives:
        x = df if hl == HL0 else make_df(hl)
        pr = S.walk_forward(x, ours(BASE), BASE, args.test)
        s_ = summarise(pr)
        rows.append({"halflife_days": hl, "accuracy": s_["accuracy"], "auc": s_["auc"], "log_loss": s_["log_loss"],
                     "confident_accuracy": s_["confident_accuracy"], "coverage": s_["coverage"], "n": s_["n"]})
        print(f"  half-life {hl:g}: accuracy {s_['accuracy']:.3f}  [{time.time() - t0:.0f}s]")
    hm = pd.DataFrame(rows)
    hm.to_csv(config.REPORTS / f"{OUT}halflife_m8.csv", index=False)
    print("\n=== 3. Rating half-life: effect on signing-success model (headline setting = %g) ===" % HL0)
    print(hm.round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. rating half-life, RQ1 (slow)
if "halflife_rq1" in args.parts:
    G, STN = d["games"], d["stints"]
    elo = match.elo_gaps(G)
    eval_comps = comps if args.synthetic else config.TOP_LEAGUES
    seasons = [2018, 2019, 2020, 2021] if args.synthetic else [2019, 2020, 2021, 2022, 2023, 2024]
    lam = rkw["lam"]

    def season_slice(g_, s_):
        c, e = pd.Timestamp(f"{s_}-08-01"), pd.Timestamp(f"{s_ + 1}-07-31")
        return g_[(g_.date >= c) & (g_.date <= e) & g_.competition_id.isin(eval_comps)].dropna(subset=["S_home", "S_away"])

    rows = []
    for hl in args.halflives:
        st = match.honest_strength(G, STN, range(seasons[0] - 3, seasons[-1] + 1),
                                   lambda cut: PlusMinus(lam=lam, halflife_days=hl).fit(G, STN, cut).ratings)
        g = G.merge(st, on="game_id", how="left").merge(elo, on="game_id")
        res = {"Player model": ([], []), "Hybrid": ([], []), "Elo": ([], [])}
        for season in seasons:
            cut = pd.Timestamp(f"{season}-08-01")
            test = season_slice(g, season)
            y = match.outcome_label(test)
            for nm, mdl in [("Player model", match.PlayerPoisson()), ("Hybrid", match.HybridPoisson()), ("Elo", match.Elo())]:
                res[nm][0].append(mdl.fit(g, cut).predict(test)); res[nm][1].append(y)
        row = {"halflife_days": hl}
        for nm, (P, Y) in res.items():
            row[f"{nm}_log_loss"] = match.metrics(np.vstack(P), np.concatenate(Y))["log_loss"]
        rows.append(row)
        print(f"  half-life {hl:g}: hybrid {row['Hybrid_log_loss']:.4f}, player {row['Player model_log_loss']:.4f}, "
              f"Elo {row['Elo_log_loss']:.4f}  [{time.time() - t0:.0f}s]")
    hr = pd.DataFrame(rows)
    hr.to_csv(config.REPORTS / f"{OUT}halflife_rq1.csv", index=False)
    print("\n=== 4. Rating half-life: effect on match prediction (pooled test log loss, lower is better) ===")
    print(hr.round(4).to_string(index=False))

print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv")
