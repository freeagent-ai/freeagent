"""M6 - Valuation: is the fee fair for the league points the player is expected to add?

Step 1  Market price of a point (calibrated from club data, no football judgement):
    across club-seasons, regress  points - league average  on  squad market value - league average,
    using each club's own changes over time (club fixed effects) so that "rich clubs win more" does not
    inflate the price (see slope_from_panel; the older between-club version was about 5 times higher).
    slope b = points per EUR 1M of squad value.  A fee F therefore claims  m(F) = b * F  points
    ("what the market says this price tag buys").
Step 2  Our estimate: the Monte Carlo draws of points added by this signing (M5), with uncertainty.
Step 3  Net value in points:  N = gain - m(F).   With tolerance tau (1 point, fixed in advance):
        P(Overpaid)  = P(N < -tau)       P(Best Buy) = P(N > +tau)       P(Fair) = the rest.
    The verdict is the most probable class, reported with its probability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from . import simulate as M

TOLERANCE_POINTS = 1.0
LABELS = ["Overpaid", "Fair", "Best Buy"]


def squad_values(games, stints, players, valuations, season: int, min_share: float = 0.15) -> pd.Series:
    """Sum of market values (EUR M) of each club's regular players, as known on 1 August of `season`."""
    sq = M.baseline_squad(games, stints, pd.Series(dtype=float), players, season)
    sq = sq[sq.share >= min_share][["club_id", "player_id"]].drop_duplicates()
    v = valuations.dropna(subset=["market_value_in_eur"]).sort_values("date")
    q = sq[["player_id"]].drop_duplicates().assign(when=pd.Timestamp(f"{season}-08-01")).sort_values("when")
    r = pd.merge_asof(q, v[["player_id", "date", "market_value_in_eur"]], left_on="when", right_on="date",
                      by="player_id", allow_exact_matches=False)
    val = sq.merge(r[["player_id", "market_value_in_eur"]], on="player_id")
    return (val.groupby("club_id").market_value_in_eur.sum() / 1e6).rename("value_m")


def slope_from_panel(x: pd.DataFrame, within_club: bool = True, n_boot: int = 300) -> dict:
    """Points per EUR 1M of squad value from a club-season panel with columns
    club_id, league, season, value_m, points.

    Both versions first remove each league-season's average (compare clubs only with their own league
    in the same year). between clubs: rich clubs win more for many reasons besides one signing, so the
    slope is inflated (about 6.9 points per EUR 100M in our data). within_club=True additionally removes
    each club's own average over time (club fixed effects): only changes in a club's squad value are
    compared with changes in its points, which strips out 'rich clubs are just better' (about 1.4)."""
    x = x.copy()
    g = x.groupby(["league", "season"])
    x["dv"] = x.value_m - g.value_m.transform("mean")
    x["dp"] = x.points - g.points.transform("mean")
    if within_club:
        x["dv"] -= x.groupby("club_id").dv.transform("mean")
        x["dp"] -= x.groupby("club_id").dp.transform("mean")
    slope = lambda d: float((d.dv * d.dp).sum() / max((d.dv ** 2).sum(), 1e-12))
    b = slope(x)
    rng = np.random.default_rng(config.SEED)
    groups = {c: d for c, d in x.groupby("club_id")}
    ids = np.array(list(groups))
    boots = [slope(pd.concat([groups[c] for c in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    return {"b": b, "se": float(np.std(boots)), "corr": float(np.corrcoef(x.dv, x.dp)[0, 1]), "n": len(x),
            "within_club": within_club}


def points_per_million(games, stints, players, valuations, club_table: pd.DataFrame, seasons,
                       within_club: bool = True) -> dict:
    """Slope b (points per EUR 1M) from club-seasons in `seasons` (see slope_from_panel)."""
    rows = []
    for y in seasons:
        v = squad_values(games, stints, players, valuations, y)
        t = club_table[club_table.season == y].set_index("club_id")
        x = pd.DataFrame({"value_m": v}).join(t[["ppg", "league", "n_games"]], how="inner").assign(season=y)
        x["points"] = x.ppg * x.n_games
        rows.append(x.rename_axis("club_id").reset_index())
    return slope_from_panel(pd.concat(rows), within_club)


def verdict_probs(gain_draws: np.ndarray, fee_eur: float, b_per_million: float,
                  tolerance: float = TOLERANCE_POINTS) -> dict:
    """Probabilities of Overpaid / Fair / Best Buy from the Monte Carlo draws of points added."""
    claimed = b_per_million * fee_eur / 1e6
    net = gain_draws - claimed
    p = np.array([(net < -tolerance).mean(), (np.abs(net) <= tolerance).mean(), (net > tolerance).mean()])
    return {"claimed_points": claimed, "net_mean": float(net.mean()), "p_overpaid": p[0], "p_fair": p[1],
            "p_best_buy": p[2], "verdict": LABELS[int(p.argmax())], "confidence": float(p.max())}


QUANTILE_LEVELS = np.linspace(0, 100, 41)          # every 2.5th percentile of the simulated points gain


def verdict_from_quantiles(q: np.ndarray, fee_eur: float, b_per_million: float,
                           tolerance: float = TOLERANCE_POINTS) -> dict:
    """Same verdict as verdict_probs, from the stored quantiles of the gain distribution, so the price of a
    point can be changed without re-running the simulation. F(x) = P(gain <= x) by linear interpolation."""
    claimed = b_per_million * fee_eur / 1e6
    probs = QUANTILE_LEVELS / 100.0
    F = lambda x: float(np.interp(x, q, probs, left=0.0, right=1.0))
    p_over = F(claimed - tolerance)
    p_best = 1.0 - F(claimed + tolerance)
    p = np.array([p_over, max(1.0 - p_over - p_best, 0.0), p_best])
    return {"claimed_points": claimed, "p_overpaid": p[0], "p_fair": p[1], "p_best_buy": p[2],
            "verdict": LABELS[int(p.argmax())], "confidence": float(p.max())}
