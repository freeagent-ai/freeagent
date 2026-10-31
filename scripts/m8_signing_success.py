"""M8: will a big-money summer signing become a regular (>= 50% of league minutes) in season one?

Protocol (no peeking at test seasons):
  1. Validation: walk-forward on seasons 2017-2018 (train on earlier seasons) picks the
     feature set and tree depth for our ensemble, by log loss.
  2. Test: walk-forward on seasons 2019-2025 with everything frozen. For each test season Y
     the models train on signings from seasons before Y only.
Reported: accuracy on all signings, and accuracy on confident predictions (p >= 0.6 or
p <= 0.4) with the share of signings that are confident (coverage).

Usage:
  python scripts/m8_signing_success.py              # real data: data/freeagent_subset.zip
Outputs (reports/): m8_validation.csv, m8_summary.csv, m8_by_season.csv,
                    m8_predictions.csv, m8_importance.csv
"""
import argparse
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
args = p.parse_args()
t0 = time.time()
VALID = [2017, 2018]
MARGIN = S.CONFIDENCE_MARGIN
GB = dict(learning_rate=0.03, max_iter=150)

tables = data.load_tables(args.data)
lt = S.league_tables(tables)
sig = S.build_signings(tables, lt)
d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
ratings = S.fit_ratings(d["games"], d["stints"], range(2015, max(args.test) + 1))
df = S.build_features(sig, tables, lt, ratings)
df = df[df.season >= 2015].reset_index(drop=True)
print(f"{len(df)} signings (fee >= EUR 5M, summer, top-5 clubs), seasons {df.season.min()}-{df.season.max()}, "
      f"base rate regular = {df.regular.mean():.3f}  [{time.time() - t0:.0f}s]")
print(df.groupby("season").agg(signings=("regular", "size"), regular_rate=("regular", "mean")).round(3).to_string())

BASE = S.NUMERIC + S.CATEGORICAL
WITH_VALUE = BASE + S.VALUE_FEATURES
NO_RATING = [f for f in BASE if f not in S.RATING_FEATURES]
MARKET = ["log_fee", "log_value", "age"]


def confident(pr):
    hit = ((pr.p >= 0.5).astype(int) == pr.regular).to_numpy()
    m = ((pr.p - 0.5).abs() >= MARGIN).to_numpy()
    return {"confident_accuracy": float(hit[m].mean()) if m.any() else np.nan, "coverage": float(m.mean())}


# ---- 1. validation: choose feature set and depth (log loss on 2017-2018)
val = []
for fname, feats in [("base", BASE), ("base + value features", WITH_VALUE)]:
    for depth in (2, 3):
        pr = S.walk_forward(df, lambda: S.Ensemble(feats, max_depth=depth, **GB), feats, VALID)
        val.append({"features": fname, "max_depth": depth,
                    **S.scores(pr.p.to_numpy(), pr.regular.to_numpy()), **confident(pr)})
val = pd.DataFrame(val).sort_values("log_loss").reset_index(drop=True)
print(f"\nValidation (seasons {VALID}), ensemble settings by log loss:")
print(val.round(3).to_string())
FEATS = WITH_VALUE if val.features[0] != "base" else BASE
DEPTH = int(val.max_depth[0])
print(f"Chosen: features = {val.features[0]}, max_depth = {DEPTH}")

# ---- 2. test: everything frozen
NAME = "Ensemble: logistic + boosting (ours)"
models = {
    "Majority class (base rate)": (lambda: S.Majority(), FEATS),
    "Market only (fee, value, age)": (lambda: S.make_model("logit", MARKET, C=1.0), MARKET),
    "Logistic regression": (lambda: S.make_model("logit", FEATS, C=0.03), FEATS),
    "Gradient boosting": (lambda: S.make_model("gb", FEATS, max_depth=DEPTH, **GB), FEATS),
    "Ours without plus-minus rating": (lambda: S.Ensemble(NO_RATING, max_depth=DEPTH, **GB), NO_RATING),
    "Ours + value features (rejected)": (lambda: S.Ensemble(WITH_VALUE, max_depth=DEPTH, **GB), WITH_VALUE),
    NAME: (lambda: S.Ensemble(FEATS, max_depth=DEPTH, **GB), FEATS),
}
preds = {name: S.walk_forward(df, fn, feats, args.test) for name, (fn, feats) in models.items()}

rows, by_season = [], []
for name, pr in preds.items():
    rows.append({"model": name, **S.scores(pr.p.to_numpy(), pr.regular.to_numpy()), **confident(pr)})
    for y, g in pr.groupby("season"):
        by_season.append({"model": name, "season": y, **S.scores(g.p.to_numpy(), g.regular.to_numpy()),
                          **confident(g)})
summary = pd.DataFrame(rows).set_index("model").sort_values("log_loss")
by_season = pd.DataFrame(by_season)

out = config.REPORTS
out.mkdir(exist_ok=True)
val.to_csv(out / "m8_validation.csv", index=False)
summary.to_csv(out / "m8_summary.csv")
by_season.to_csv(out / "m8_by_season.csv", index=False)
ours = preds[NAME]
ours[["player_name", "transfer_date", "season", "from_club_name", "to_club_name", "to_league", "transfer_fee",
      "share", "regular", "p"]].to_csv(out / "m8_predictions.csv", index=False)

rng = np.random.default_rng(config.SEED)


def hits(pr, only_confident=False):
    h = ((pr.p >= 0.5).astype(int) == pr.regular).to_numpy().astype(float)
    return h[((pr.p - 0.5).abs() >= MARGIN).to_numpy()] if only_confident else h


def ci(h):
    b = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])


def paired(a, b):
    diff = hits(preds[a]) - hits(preds[b])
    bt = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bt, [2.5, 97.5])
    v = "significantly better" if lo > 0 else "significantly worse" if hi < 0 else "no significant difference"
    return f"{a} vs {b}: accuracy {diff.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}) -> {v}"


print(f"\n=== M8 test results, seasons {args.test[0]}-{args.test[-1]} (lower log loss is better) ===")
print(summary.round(3).to_string())
a_lo, a_hi = ci(hits(ours))
c_lo, c_hi = ci(hits(ours, True))
r = summary.loc[NAME]
print(f"\nOur accuracy, all signings:       {r.accuracy:.3f} (95% CI {a_lo:.3f} to {a_hi:.3f}), n = {int(r.n)}")
print(f"Our accuracy, confident signings: {r.confident_accuracy:.3f} (95% CI {c_lo:.3f} to {c_hi:.3f}), "
      f"coverage = {r.coverage:.1%} of signings")
for b in ["Majority class (base rate)", "Market only (fee, value, age)", "Ours without plus-minus rating"]:
    print(paired(NAME, b))
print("\nBy season (ours):")
print(by_season[by_season.model == NAME][["season", "accuracy", "confident_accuracy", "coverage", "auc", "n"]]
      .round(3).to_string(index=False))

# ---- which features matter: permutation importance on the last test season
last = args.test[-1]
trn, tst = df[df.season < last], df[df.season == last]
m = S.Ensemble(FEATS, max_depth=DEPTH, **GB).fit(trn[FEATS], trn.regular)
imp = S.permutation_importance(m, tst[FEATS], tst.regular).rename("log_loss_increase")
imp.to_csv(out / "m8_importance.csv")
print(f"\nTop features (permutation importance, season {last}):")
print(imp.head(10).round(4).to_string())
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/m8_*.csv")
