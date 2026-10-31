"""M2 - ML prior for player ratings.

Problem: a player with few minutes gets a rating close to 0 (the ridge shrinks him to the
league average), even if his stats say he is clearly good or clearly poor.

Fix: learn mu_i = f(stats_i), a "box-score" guess of the rating, and shrink each player
toward mu_i instead of toward 0 (M3 already supports this: beta ~ N(mu, sigma^2 / lam)).

  1. At cutoff c, fit M3 with no prior. Players with >= MIN_TARGET_MINUTES are reliable:
     their ratings are the training targets.
  2. Features per player, from matches before c only (time-decayed): minutes, minutes per
     appearance (starter or sub), goals / assists / cards per 90, share of minutes in the
     Champions League / Europa League, age at c, position. NOT market value (that would
     smuggle in the transfer market we are trying to judge) and NOT team results (that would
     put the team effect back into the player).
  3. Gradient-boosted regression, weighted by minutes, predicts mu for every player.
  4. Refit M3 with prior = mu.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .impact import PlusMinus

MIN_TARGET_MINUTES = 2000
FEATURES = ["log_minutes", "mins_per_app", "goals90", "assists90", "yellow90", "red90",
            "euro_share", "age", "position"]


def player_stats(appearances: pd.DataFrame, players: pd.DataFrame, cutoff, halflife_days: float = 1460.0):
    """Time-decayed per-player stats from appearances before `cutoff`."""
    cut = pd.Timestamp(cutoff)
    a = appearances[appearances.date < cut]
    w = 0.5 ** ((cut - a.date).dt.days.to_numpy(float) / halflife_days)
    df = pd.DataFrame({"player_id": a.player_id.to_numpy(), "w": w,
                       "mins": w * a.minutes_played.to_numpy(), "app": w * (a.minutes_played.to_numpy() > 0),
                       "goals": w * a.goals.to_numpy(), "assists": w * a.assists.to_numpy(),
                       "yellow": w * a.yellow_cards.to_numpy(), "red": w * a.red_cards.to_numpy(),
                       "euro": w * a.minutes_played.to_numpy() * a.competition_id.isin(config.CUPS).to_numpy()})
    g = df.groupby("player_id").sum()
    m = g.mins.clip(lower=1)
    out = pd.DataFrame({"log_minutes": np.log1p(g.mins), "mins_per_app": g.mins / g.app.clip(lower=1e-9),
                        "goals90": 90 * g.goals / m, "assists90": 90 * g.assists / m,
                        "yellow90": 90 * g.yellow / m, "red90": 90 * g.red / m, "euro_share": g.euro / m})
    p = players.set_index("player_id")
    out["age"] = (cut - out.index.map(p.date_of_birth)).days / 365.25
    out["position"] = pd.Categorical(out.index.map(p.position).fillna("Unknown"))
    return out


class Prior:
    """Fits mu = f(stats) at a cutoff and returns it as a Series indexed by player_id."""
    def __init__(self, lam: float = 100.0, halflife_days: float = 1460.0, seed: int = config.SEED):
        self.lam, self.halflife_days, self.seed = lam, halflife_days, seed

    def fit(self, games, stints, appearances, players, cutoff) -> "Prior":
        from sklearn.ensemble import HistGradientBoostingRegressor
        base = PlusMinus(lam=self.lam, halflife_days=self.halflife_days).fit(games, stints, cutoff)
        X = player_stats(appearances, players, cutoff, self.halflife_days)
        tgt = pd.DataFrame({"y": base.ratings, "mins": base.minutes})
        tgt = tgt[tgt.mins >= MIN_TARGET_MINUTES].join(X, how="inner")
        self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   min_samples_leaf=40, l2_regularization=1.0,
                                                   categorical_features="from_dtype", random_state=self.seed)
        self.model.fit(tgt[FEATURES], tgt.y, sample_weight=np.sqrt(tgt.mins))
        self.mu = pd.Series(self.model.predict(X[FEATURES]), index=X.index, name="prior")
        self.n_targets, self.base = len(tgt), base
        return self

    def ratings(self, games, stints, cutoff) -> pd.Series:
        """M3 ratings shrunk toward this prior."""
        return PlusMinus(lam=self.lam, halflife_days=self.halflife_days).fit(
            games, stints, cutoff, prior=self.mu).ratings
