"""Option B: will a player's market value RISE over the next 12 months?

FAIR-TEST RULES (fixed before running):
  * One row per player per season (cutoff 1 August), players with >= 900 top-5-league minutes last season.
  * Label: latest market value 6-12 months after the cutoff is higher than the latest value before it.
  * Walk-forward: each test season is predicted by models trained on earlier seasons only. Test 2019-2024.
  * No tuning: every model uses fixed default settings written before running.
  * Baselines: majority class; AGE ONLY (young players rise, older ones fall, the obvious rule);
    age + recent value trend; age + current value. Paired bootstrap of ours minus each.

Usage: python scripts/value_rise.py               # real data
       python scripts/value_rise.py --synthetic   # quick check on the fake world (fake market values)
Outputs (reports/): value_rise_summary.csv, value_rise_pairwise.csv, value_rise_predictions.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, success as S, value_rise as VR

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(config.DATA / "freeagent_subset.zip"))
p.add_argument("--test", nargs="+", type=int, default=list(range(2019, 2025)))
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()
t0 = time.time()
rng = np.random.default_rng(config.SEED)

if args.synthetic:
    from freeagent.synthetic import make_world, make_valuations
    w = make_world(n_leagues=3, seasons=tuple(range(2014, 2022)), seed=5)
    w["player_valuations"] = make_valuations(w, seed=5)
    tables = data.standardise(w)
    d = data.prepare(tables, competitions=sorted(tables["games"].competition_id.unique()), first_season=2014)
    args.test, rkw, OUT = [2019, 2020], dict(lam=20, halflife_days=730), "value_rise_synthetic_"
else:
    tables = data.load_tables(args.data)
    d = data.prepare(tables, competitions=config.RATING_COMPETITIONS)
    rkw, OUT = {}, "value_rise_"
lt = S.league_tables(tables)
ratings = S.fit_ratings(d["games"], d["stints"], range(2015, max(args.test) + 1), **rkw)
df = VR.build_panel(tables, lt, range(2015, max(args.test) + 1), ratings)
print(f"{len(df)} player-seasons, {df.season.nunique()} seasons, base rate (value rises) = {df.rise.mean():.3f}  "
      f"[{time.time() - t0:.0f}s]")
print(df.groupby("season").agg(players=("rise", "size"), rise_rate=("rise", "mean")).round(3).to_string())

usable = lambda fs: [f for f in fs if df[f].notna().any()]
ALL = usable(VR.NUMERIC + VR.CATEGORICAL)
AGE = ["age", "age_sq"]
AGE_TREND = usable(["age", "age_sq", "value_trend"])
AGE_VALUE = usable(["age", "age_sq", "log_value"])
OURS = "OURS: logistic + boosting ensemble (all features)"
models = {
    "Majority class": (lambda: S.Majority(), AGE),
    "Age only": (lambda: S.make_model("logit", AGE, C=1.0), AGE),
    "Age + value trend": (lambda: S.make_model("logit", AGE_TREND, C=1.0), AGE_TREND),
    "Age + current value": (lambda: S.make_model("logit", AGE_VALUE, C=1.0), AGE_VALUE),
    "Logistic regression (all features)": (lambda: S.make_model("logit", ALL, C=0.03), ALL),
    "Gradient boosting (all features)": (lambda: S.make_model("gb", ALL, max_depth=3, max_iter=150, learning_rate=0.03), ALL),
    OURS: (lambda: S.Ensemble(ALL, max_depth=3, max_iter=150, learning_rate=0.03), ALL),
}
preds = {n: VR.walk_forward(df, fn, f, args.test) for n, (fn, f) in models.items()}
y = preds[OURS].rise.to_numpy()
hit = {n: ((pr.p >= 0.5).astype(int) == pr.rise).to_numpy().astype(float) for n, pr in preds.items()}
rows = []
for n, pr in preds.items():
    sc = S.scores(pr.p.to_numpy(), pr.rise.to_numpy())
    m = (pr.p - 0.5).abs().to_numpy() >= S.CONFIDENCE_MARGIN
    rows.append({"model": n, **sc, "confident_accuracy": hit[n][m].mean() if m.any() else np.nan,
                 "coverage": m.mean()})
summ = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
summ.to_csv(config.REPORTS / f"{OUT}summary.csv")
print(f"\n=== TEST seasons {args.test[0]}-{args.test[-1]}: will the market value rise in the next 12 months? ===")
print(summ[["accuracy", "balanced_accuracy", "auc", "log_loss", "brier", "confident_accuracy", "coverage", "n"]]
      .round(3).to_string())


def boot(diff, n=2000):
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


pair = []
for n in preds:
    if n == OURS:
        continue
    diff = hit[OURS] - hit[n]
    lo, hi = boot(diff)
    pair.append({"versus": n, "accuracy_gain": diff.mean(), "lo": lo, "hi": hi,
                 "verdict": "OURS BETTER" if lo > 0 else "ours worse" if hi < 0 else "tied"})
pair = pd.DataFrame(pair).set_index("versus")
pair.to_csv(config.REPORTS / f"{OUT}pairwise.csv")
print("\nOurs minus each model, accuracy (95% CI):")
print(pair.round(3).to_string())
lo, hi = boot(hit[OURS])
print(f"\nOur accuracy: {hit[OURS].mean():.3f} (95% CI {lo:.3f} to {hi:.3f}), n = {len(y)}")
print("\nBy season (ours vs age only vs majority):")
bs = pd.DataFrame({n: preds[n].assign(h=hit[n]).groupby("season").h.mean() for n in [OURS, "Age only", "Majority class"]})
print(bs.round(3).to_string())
preds[OURS][["player_id", "season", "age", "value_now", "value_future", "rise", "p"]].to_csv(
    config.REPORTS / f"{OUT}predictions.csv", index=False)
print(f"\nTotal time {time.time() - t0:.0f}s. Saved reports/{OUT}*.csv")
