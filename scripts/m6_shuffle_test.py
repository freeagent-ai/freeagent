"""M6 control: do our verdicts carry information beyond the fee?  (run at BOTH price levels)

Uses the stored distribution of each signing's points gain (41 quantiles), so verdicts are recomputed exactly
as in the pipeline.
  1. Fee-only rule: how much of the verdict can a one-line rule on the fee reproduce, with no model?
  2. Shuffle test: swap whole gain distributions between signings of the same season, keep each signing's
     fee, recompute the verdicts. Verdicts that do not change were decided by the fee, not by our estimate.
  3. Players with and without a rating.
Usage: python scripts/m6_shuffle_test.py       Output: reports/m6_shuffle_test.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, value as V

o = pd.read_csv(config.REPORTS / "m5_signing_points.csv")
Q = o[[f"gq_{i:02d}" for i in range(len(V.QUANTILE_LEVELS))]].to_numpy()
fee = o.transfer_fee.to_numpy()
PRICES = {"LOW (club fixed effects)": ("price_b_per_100m", "verdict", "p_overpaid"),
          "HIGH (between clubs)": ("price_high_per_100m", "verdict_high", "p_overpaid_high")}


def verdicts(Qm, b):
    return np.array([V.verdict_from_quantiles(q, f, b)["verdict"] for q, f in zip(Qm, fee)])


rng = np.random.default_rng(config.SEED)
rows = []
for tag, (bcol, vcol, pcol) in PRICES.items():
    b = o[bcol].iloc[0] / 100.0
    v0 = verdicts(Q, b)
    print(f"\n===== {tag}: {100 * b:.2f} points per EUR 100M =====")
    print(f"Check: recomputed verdicts match the stored ones for {np.mean(v0 == o[vcol]):.1%} of signings")
    print("Verdict mix: " + ", ".join(f"{k} {np.mean(v0 == k):.0%}" for k in V.LABELS))
    rho = spearmanr(fee, o[pcol])[0]
    best_acc, best_t = 0, None
    for t in np.linspace(5e6, 120e6, 116):
        for hi in ("Overpaid",):
            acc = np.mean(np.where(fee >= t, hi, "Fair") == v0)
            if acc > best_acc:
                best_acc, best_t = acc, t
    allfair = np.mean(v0 == "Fair")
    print(f"1. Spearman(fee, P(Overpaid)) = {rho:+.3f}")
    print(f"   Best fee-only rule 'Overpaid if fee >= EUR {best_t / 1e6:.0f}M, else Fair' reproduces {best_acc:.1%} of verdicts;"
          f" always saying 'Fair' reproduces {allfair:.1%}")
    same = []
    for _ in range(100):
        Qs = Q.copy()
        for _, idx in o.groupby("season").groups.items():
            Qs[idx] = Q[rng.permutation(np.asarray(idx))]
        same.append(np.mean(verdicts(Qs, b) == v0))
    lo, hi = np.percentile(same, [2.5, 97.5])
    print(f"2. Shuffle: {np.mean(same):.1%} of verdicts unchanged when estimates are shuffled between signings "
          f"(95% range {lo:.1%} to {hi:.1%})")
    print(f"   => our own estimates decide about {1 - np.mean(same):.0%} of verdicts")
    rows += [{"price": tag, "test": "fee_only_rule_agreement", "value": best_acc},
             {"price": tag, "test": "all_fair_agreement", "value": allfair},
             {"price": tag, "test": "shuffle_unchanged", "value": np.mean(same)},
             {"price": tag, "test": "spearman_fee_p_overpaid", "value": rho}]
    g = o.assign(verdict_now=v0, known=np.where(o.known_player == 1, "has rating", "no rating (prior only)"))
    tab = g.groupby("known").agg(n=("verdict_now", "size"), fee_m=("transfer_fee", lambda x: x.mean() / 1e6),
                                 over=("verdict_now", lambda x: (x == "Overpaid").mean()),
                                 fair=("verdict_now", lambda x: (x == "Fair").mean()),
                                 best=("verdict_now", lambda x: (x == "Best Buy").mean()))
    print("3. By whether the player had a rating:")
    print(tab.round(3).to_string())
pd.DataFrame(rows).to_csv(config.REPORTS / "m6_shuffle_test.csv", index=False)
print("\nSaved reports/m6_shuffle_test.csv")
