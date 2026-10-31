"""M8 model comparison: is our signing-success model better than the standard model families?

FAIR-TEST RULES (fixed before running):
  * Every model sees the SAME features (the round-1 base features) and the SAME walk-forward test.
  * Every competitor gets the SAME small tuning budget on validation seasons 2017-2018 (chosen by log loss).
    Ours is the frozen round-1 ensemble (chosen earlier on the same validation seasons).
  * Test seasons 2019-2025 are scored once. No tuning on them.
  * Paired bootstrap (resampling the same signings) gives a 95% CI for ours-minus-theirs on accuracy, log loss,
    Brier score and AUC, with the sign flipped so that POSITIVE always means "ours is better".
  * Honest wording: we compare with standard and published MODEL FAMILIES on our data. Commercial systems are not public.

Usage: python scripts/m8_compare.py               # real data
       python scripts/m8_compare.py --synthetic   # quick check on the fake world
Outputs (reports/): m8_compare_validation.csv, m8_compare_test.csv, m8_compare_pairwise.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, success as S

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--test", nargs="+", type=int, default=list(range(2019, 2026)))
p.add_argument("--boot", type=int, default=1000)
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()
t0 = time.time()
VALID = [2017, 2018]
rng = np.random.default_rng(config.SEED)
OUT = "m8_compare_synthetic_" if args.synthetic else "m8_compare_"

if args.synthetic:
    from freeagent.synthetic import make_world
    tables = data.standardise(make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5))
    args.test, args.boot = [2019, 2020, 2021], 100
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
df = df[df.season >= 2015].reset_index(drop=True)
usable = lambda fs: [f for f in fs if df[f].notna().any()]
BASE = usable(S.NUMERIC + S.CATEGORICAL)
MARKET = usable(["log_fee", "log_value", "age"])
print(f"{len(df)} signings, base rate regular = {df.regular.mean():.3f}  [{time.time() - t0:.0f}s]")

# ---------------------------------------------------------------- candidate families and their (small) tuning grids
grids = {
    "Logistic regression": [dict(C=c) for c in (0.01, 0.03, 0.1, 1.0)],
    "Gradient boosting": [dict(depth=dp, it=it) for dp in (2, 3) for it in (150, 300)],
    "Random forest": [dict(leaf=l) for l in (3, 10, 30)],
    "Extra trees": [dict(leaf=l) for l in (3, 10, 30)],
    "AdaBoost": [dict(n=n, lr=lr) for n in (100, 300) for lr in (0.1, 0.5)],
    "Naive Bayes": [dict()],
    "k-nearest neighbours": [dict(k=k) for k in ((5, 10, 20) if args.synthetic else (25, 50, 100))],
    "Support vector machine": [dict(C=c) for c in (0.1, 1.0, 10.0)],
    "Neural network (MLP)": [dict(hidden=h, alpha=a) for h in ((16,), (32, 16)) for a in (1e-2, 1.0)],
}
try:
    import xgboost  # noqa: F401
    grids["XGBoost"] = [dict(n=n, depth=dp) for n in (150, 300) for dp in (2, 3)]
except ImportError:
    print("(xgboost not installed: skipped)")


def build(family, kw):
    if family == "Logistic regression":
        return lambda: S.make_model("logit", BASE, C=kw["C"])
    if family == "Gradient boosting":
        return lambda: S.make_model("gb", BASE, max_depth=kw["depth"], max_iter=kw["it"], learning_rate=0.03)
    kind = {"Random forest": "rf", "Extra trees": "et", "AdaBoost": "ada", "Naive Bayes": "nb",
            "k-nearest neighbours": "knn", "Support vector machine": "svm", "Neural network (MLP)": "mlp",
            "XGBoost": "xgb"}[family]
    return lambda: S.make_panel_model(kind, BASE, **kw)


def ll_row(pr):
    p_ = np.clip(pr.p.to_numpy(), 1e-6, 1 - 1e-6)
    return -(pr.regular.to_numpy() * np.log(p_) + (1 - pr.regular.to_numpy()) * np.log(1 - p_))


val_rows, chosen = [], {}
for fam, glist in grids.items():
    best = None
    for kw in glist:
        pr = S.walk_forward(df, build(fam, kw), BASE, VALID)
        ll = float(ll_row(pr).mean())
        val_rows.append({"family": fam, "settings": str(kw), "val_log_loss": ll,
                         "val_accuracy": float(((pr.p >= 0.5).astype(int) == pr.regular).mean())})
        if best is None or ll < best[0]:
            best = (ll, kw)
    chosen[fam] = best[1]
    print(f"  tuned {fam}: {best[1]}  (validation log loss {best[0]:.3f})  [{time.time() - t0:.0f}s]")
val = pd.DataFrame(val_rows)
val.to_csv(config.REPORTS / f"{OUT}validation.csv", index=False)

# ---------------------------------------------------------------- test (frozen)
OURS = "OURS: logistic + boosting ensemble"
models = {"Majority class": (lambda: S.Majority(), BASE),
          "Market only (fee, value, age)": (lambda: S.make_model("logit", MARKET, C=1.0), MARKET),
          OURS: (lambda: S.Ensemble(BASE, max_depth=3, max_iter=150, learning_rate=0.03), BASE)}
for fam in grids:
    models[fam] = (build(fam, chosen[fam]), BASE)
preds = {n: S.walk_forward(df, fn, f, args.test) for n, (fn, f) in models.items()}
ref = preds[OURS]
for n, pr in preds.items():
    assert (pr.index == ref.index).all(), "predictions are not aligned"
y = ref.regular.to_numpy()


def metrics_rows(pr):
    pp = np.clip(pr.p.to_numpy(), 1e-6, 1 - 1e-6)
    return {"acc": ((pp >= 0.5).astype(int) == y).astype(float), "ll": ll_row(pr), "brier": (pp - y) ** 2}


M = {n: metrics_rows(pr) for n, pr in preds.items()}
P = {n: np.clip(pr.p.to_numpy(), 1e-6, 1 - 1e-6) for n, pr in preds.items()}
conf = lambda pr: ((pr.p - 0.5).abs() >= S.CONFIDENCE_MARGIN).to_numpy()
rows = []
for n, pr in preds.items():
    m = conf(pr)
    hit = M[n]["acc"]
    rows.append({"model": n, "accuracy": hit.mean(), "log_loss": M[n]["ll"].mean(), "brier": M[n]["brier"].mean(),
                 "auc": roc_auc_score(y, P[n]) if np.std(P[n]) > 0 else 0.5,
                 "confident_accuracy": hit[m].mean() if m.any() else np.nan, "coverage": m.mean(), "n": len(y)})
test = pd.DataFrame(rows).set_index("model")
test.sort_values("log_loss").to_csv(config.REPORTS / f"{OUT}test.csv")
print(f"\n=== TEST seasons {args.test[0]}-{args.test[-1]}, walk-forward, frozen (sorted by log loss, lower is better) ===")
print(test.sort_values("log_loss").round(3).to_string())

# ---------------------------------------------------------------- paired bootstrap: ours vs each
idx = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
bt_auc = {n: np.array([roc_auc_score(y[i], P[n][i]) if len(np.unique(y[i])) > 1 and np.std(P[n][i]) > 0 else 0.5
                       for i in idx]) for n in preds}
pair = []
for n in preds:
    if n == OURS:
        continue
    rec = {"versus": n}
    for key, label, sign in [("acc", "accuracy", +1), ("ll", "log_loss", -1), ("brier", "brier", -1)]:
        diff = sign * (M[OURS][key] - M[n][key])
        bt = np.array([diff[i].mean() for i in idx])
        lo, hi = np.percentile(bt, [2.5, 97.5])
        rec[f"{label}_adv"], rec[f"{label}_lo"], rec[f"{label}_hi"] = diff.mean(), lo, hi
    bt = bt_auc[OURS] - bt_auc[n]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    rec["auc_adv"], rec["auc_lo"], rec["auc_hi"] = test.loc[OURS, "auc"] - test.loc[n, "auc"], lo, hi
    pair.append(rec)
pair = pd.DataFrame(pair).set_index("versus")


def verdict(lo, hi):
    return "BETTER" if lo > 0 else "worse" if hi < 0 else "tied"


for k in ("accuracy", "log_loss", "brier", "auc"):
    pair[f"{k}_verdict"] = [verdict(a, b) for a, b in zip(pair[f"{k}_lo"], pair[f"{k}_hi"])]
pair.to_csv(config.REPORTS / f"{OUT}pairwise.csv")
print("\n=== Ours versus each model: advantage of OURS (positive = ours better), 95% CI, verdict ===")
show = pd.DataFrame({k: [f"{pair[f'{k}_adv'][v]:+.3f} [{pair[f'{k}_lo'][v]:+.3f},{pair[f'{k}_hi'][v]:+.3f}] {pair[f'{k}_verdict'][v]}"
                         for v in pair.index] for k in ("accuracy", "log_loss", "brier", "auc")}, index=pair.index)
print(show.to_string())
print("\nScoreboard (how many of the other models ours is significantly BETTER than / tied with / worse than):")
for k in ("accuracy", "log_loss", "brier", "auc"):
    c = pair[f"{k}_verdict"].value_counts()
    print(f"  {k:9s}: better {c.get('BETTER', 0)}, tied {c.get('tied', 0)}, worse {c.get('worse', 0)} (of {len(pair)})")
print("\nRank of ours among all models (1 = best): accuracy %d, log loss %d, brier %d, auc %d  (of %d)" % (
    test.accuracy.rank(ascending=False)[OURS], test.log_loss.rank()[OURS], test.brier.rank()[OURS],
    test.auc.rank(ascending=False)[OURS], len(test)))
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv")
