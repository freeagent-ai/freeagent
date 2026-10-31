"""M8 round 2: can we lift signing-success accuracy honestly, and how do we compare with published model families?

STRICT PROTOCOL (written before running):
  * Everything is CHOSEN on validation seasons 2017-2018 only (train on 2015+ before each season):
    feature set (base vs base + 7 new features), logistic strength C, tree depth, boosting rounds,
    and the decision threshold (kept at 0.5 unless a different one gains >= 1 point on validation).
  * Then the test seasons 2019-2025 are scored ONCE with everything frozen (walk-forward, train only on the past).
  * NOTE: the original M8 run already showed us the test numbers, so a gain on the same test seasons is
    slightly optimistic. We say so in the report.
Comparison (same data, same walk-forward protocol): majority class, market-only, random forest, AdaBoost and
naive Bayes (the model families of the published transfer-success paper, re-implemented on our features),
our round-1 model and our round-2 model.
Extra table (NOT used for claims): the same models under a random 70/30 split, to show how much a
random split flatters accuracy.

Usage: python scripts/m8_improve.py                # real data, about 5-10 minutes
       python scripts/m8_improve.py --synthetic    # quick check on the fake world
Outputs (reports/): m8_improve_validation.csv, m8_improve_summary.csv, m8_improve_random_split.csv,
                    m8_improve_predictions.csv
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, success as S

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--test", nargs="+", type=int, default=list(range(2019, 2026)))
p.add_argument("--repeats", type=int, default=20, help="random-split repeats")
p.add_argument("--synthetic", action="store_true", help="quick check on the fake world (no real data needed)")
args = p.parse_args()
t0 = time.time()
VALID = [2017, 2018]
MARGIN = S.CONFIDENCE_MARGIN
rng = np.random.default_rng(config.SEED)
OUT = "m8_improve_synthetic_" if args.synthetic else "m8_improve_"   # never overwrite real results with fake ones

if args.synthetic:
    from freeagent.synthetic import make_world
    tables = data.standardise(make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5))
    args.test, args.repeats = [2019, 2020, 2021], 3
    d = data.prepare(tables, competitions=sorted(tables["games"].competition_id.unique()), first_season=2014)
    rkw = dict(lam=20, halflife_days=730)
else:
    tables = data.load_tables(args.data)
    d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
    rkw = {}
lt = S.league_tables(tables)
sig = S.build_signings(tables, lt)
ratings = S.fit_ratings(d["games"], d["stints"], range(2015, max(args.test) + 1), **rkw)
df = S.build_features(sig, tables, lt, ratings)
df = S.add_extra_features(df, tables, lt, ratings)
df = df[df.season >= 2015].reset_index(drop=True)
print(f"{len(df)} signings, base rate regular = {df.regular.mean():.3f}  [{time.time() - t0:.0f}s]")
print("New features, share of non-missing values:")
print(df[S.EXTRA].notna().mean().round(2).to_string())

def usable(fs):
    """Drop features with no values at all (only happens in the synthetic world, which has no market values)."""
    return [f for f in fs if df[f].notna().any()]


BASE = usable(S.NUMERIC + S.CATEGORICAL)
PLUS = usable(S.NUMERIC + S.CATEGORICAL + S.EXTRA)
MARKET = usable(["log_fee", "log_value", "age"])


def acc(pr, thr=0.5):
    return float(((pr.p >= thr).astype(int) == pr.regular).mean())


def confident(pr, thr=0.5):
    hit = ((pr.p >= thr).astype(int) == pr.regular).to_numpy()
    m = ((pr.p - 0.5).abs() >= MARGIN).to_numpy()
    return {"confident_accuracy": float(hit[m].mean()) if m.any() else np.nan, "coverage": float(m.mean())}


def row(pr, thr=0.5):
    sc = S.scores(pr.p.to_numpy(), pr.regular.to_numpy())
    sc["accuracy"] = acc(pr, thr)
    return {**sc, **confident(pr, thr)}


# ---------------------------------------------------------------- 1. validation grid (seasons 2017-2018 only)
GB = dict(learning_rate=0.03)
cands = {}
for fname, feats in [("base", BASE), ("base+new", PLUS)]:
    for C, depth, it in itertools.product((0.01, 0.03, 0.1), (2, 3), (150, 300)):
        cands[f"ens[{fname}] C={C} depth={depth} iter={it}"] = (
            (lambda f=feats, C=C, dp=depth, it=it: S.Ensemble(f, C=C, max_depth=dp, max_iter=it, **GB)), feats, fname)
    for C in (0.01, 0.03, 0.1):
        cands[f"logit[{fname}] C={C}"] = ((lambda f=feats, C=C: S.make_model("logit", f, C=C)), feats, fname)

val = []
valpred = {}
for name, (fn, feats, fname) in cands.items():
    pr = S.walk_forward(df, fn, feats, VALID)
    valpred[name] = pr
    val.append({"candidate": name, "features": fname, **row(pr)})
val = pd.DataFrame(val).sort_values("log_loss").reset_index(drop=True)
print(f"\nValidation (seasons {VALID}); {len(cands)} candidates, best 10 by log loss:")
print(val.head(10).round(3).to_string())
best_name = val.candidate[0]
best_fn, best_feats, best_fset = cands[best_name]
print(f"\nChosen on validation: {best_name}")

# decision threshold: only move off 0.5 if validation accuracy gains >= 1 point
pr_v = valpred[best_name]
grid = np.round(np.arange(0.44, 0.561, 0.01), 2)
va = {t: acc(pr_v, t) for t in grid}
thr = max(va, key=va.get)
THR = float(thr) if va[thr] - va[0.5] >= 0.01 else 0.5
print(f"Threshold: best on validation {thr} (acc {va[thr]:.3f} vs {va[0.5]:.3f} at 0.5) -> using {THR}")
val.to_csv(config.REPORTS / f"{OUT}validation.csv", index=False)

# ---------------------------------------------------------------- 2. test: everything frozen
R1 = "Round 1 (ours: ensemble, base features)"
R2 = "Round 2 (ours: " + best_fset + ")"
models = {
    "Majority class (base rate)": (lambda: S.Majority(), BASE),
    "Market only (fee, value, age)": (lambda: S.make_model("logit", MARKET, C=1.0), MARKET),
    "Naive Bayes (published family)": (lambda: S.make_baseline("nb", BASE), BASE),
    "AdaBoost (published family)": (lambda: S.make_baseline("ada", BASE), BASE),
    "Random forest (published family)": (lambda: S.make_baseline("rf", BASE), BASE),
    R1: (lambda: S.Ensemble(BASE, max_depth=3, max_iter=150, learning_rate=0.03), BASE),
    R2: (best_fn, best_feats),
}
preds = {n: S.walk_forward(df, fn, f, args.test) for n, (fn, f) in models.items()}
thr_of = {n: (THR if n == R2 else 0.5) for n in models}

rows, by_season = [], []
for n, pr in preds.items():
    rows.append({"model": n, **row(pr, thr_of[n])})
    for y, g in pr.groupby("season"):
        by_season.append({"model": n, "season": y, "accuracy": acc(g, thr_of[n]), "n": len(g)})
summary = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
summary.to_csv(config.REPORTS / f"{OUT}summary.csv")
pr2 = preds[R2]
pr2[["player_name", "season", "to_club_name", "transfer_fee", "share", "regular", "p"]].to_csv(
    config.REPORTS / f"{OUT}predictions.csv", index=False)


def hits(pr, thr=0.5):
    return ((pr.p >= thr).astype(int) == pr.regular).to_numpy().astype(float)


def boot(h, n=2000):
    b = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def paired(a, b):
    diff = hits(preds[a], thr_of[a]) - hits(preds[b], thr_of[b])
    bt = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    v = "significantly better" if lo > 0 else "significantly worse" if hi < 0 else "no significant difference"
    return f"  {a} vs {b}: {diff.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}) -> {v}"


print(f"\n=== TEST seasons {args.test[0]}-{args.test[-1]} (walk-forward, frozen; sorted by accuracy) ===")
print(summary[["accuracy", "balanced_accuracy", "auc", "log_loss", "confident_accuracy", "coverage", "n"]]
      .round(3).to_string())
h2 = hits(pr2, THR)
lo, hi = boot(h2)
print(f"\nRound 2 accuracy, all signings: {h2.mean():.3f} (95% CI {lo:.3f} to {hi:.3f}), n = {len(h2)}, threshold {THR}")
print(f"Round 2 confident accuracy: {summary.loc[R2].confident_accuracy:.3f} on {summary.loc[R2].coverage:.1%} of signings")
print("Paired comparisons (accuracy difference):")
for b in ["Majority class (base rate)", R1, "Random forest (published family)", "AdaBoost (published family)",
          "Naive Bayes (published family)"]:
    print(paired(R2, b))
print("\nBy season (round 2 vs round 1):")
bs = pd.DataFrame(by_season).pivot(index="season", columns="model", values="accuracy")[[R1, R2]]
print(bs.round(3).to_string())
tgt = h2.mean()
print(f"\nTarget check: overall {tgt:.1%} -> " + ("at least 67%" if tgt >= 0.67 else "BELOW 67%") +
      ("; at least 70%" if tgt >= 0.70 else ""))

# ---------------------------------------------------------------- 3. what a random split does (not used for claims)
print(f"\n=== Random 70/30 split, {args.repeats} repeats, all seasons mixed (NOT our protocol; shows the protocol effect) ===")
rs = {n: [] for n in ["Naive Bayes (published family)", "AdaBoost (published family)",
                      "Random forest (published family)", R1, R2]}
n_all = len(df)
for _ in range(args.repeats):
    idx = rng.permutation(n_all)
    cut = int(0.7 * n_all)
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    for n in rs:
        fn, feats = models[n]
        m = fn().fit(tr[feats], tr.regular)
        pt = m.predict_proba(te[feats])[:, 1]
        rs[n].append(float(((pt >= (THR if n == R2 else 0.5)).astype(int) == te.regular.to_numpy()).mean()))
rsdf = pd.DataFrame({"model": list(rs), "random_split_accuracy": [np.mean(v) for v in rs.values()],
                     "sd": [np.std(v) for v in rs.values()]}).set_index("model")
rsdf["walk_forward_accuracy"] = [summary.loc[n].accuracy for n in rsdf.index]
rsdf.to_csv(config.REPORTS / f"{OUT}random_split.csv")
print(rsdf.round(3).to_string())
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv")
