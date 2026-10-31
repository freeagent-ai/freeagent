"""M3 - Bayesian plus-minus player ratings.

Model, one equation per match m:
    GD_m = h_season(m) + sum_i  sign_i * share_i,m * beta_i  + noise
where sign = +1 for home players, -1 for away players, share = minutes / 90.

Estimate, a ridge regression pulled toward a prior mean mu (from M2) instead of zero:
    beta_hat = argmin  sum_m w_m (y_m - h - x_m.beta)^2  +  lam * ||beta - mu||^2
Closed form (set the gradient to zero):
    (Z'WZ + lam*D) [h; beta] = Z'Wy + lam*D [0; mu]        (D = 0 for h, 1 for beta)
Bayesian reading: beta ~ N(mu, sigma^2/lam), so the posterior covariance is
    Cov = sigma^2 * (Z'WZ + lam*D)^-1
which gives every rating an uncertainty. w_m = 0.5 ** (age_in_days / halflife) down-weights
old matches. Home advantage h is separate per season (Covid seasons differ).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg

DIRECT_LIMIT = 6000   # above this many unknowns use conjugate gradient (memory-safe on real data)


class _Solver:
    """Solves A x = b for the ridge system: exact LU when small, preconditioned CG when large."""
    def __init__(self, A):
        self.A = A.tocsc()
        self.lu = splu(self.A) if A.shape[0] <= DIRECT_LIMIT else None
        self.M = sp.diags(1.0 / A.diagonal())

    def solve(self, b):
        if self.lu is not None:
            return self.lu.solve(b)
        x, info = cg(self.A, b, M=self.M, rtol=1e-8, maxiter=5000)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient did not converge (info={info})")
        return x


@dataclass
class PlusMinus:
    lam: float = 20.0
    halflife_days: float = 730.0
    ratings: pd.Series = field(default=None, repr=False)
    home: pd.Series = field(default=None, repr=False)

    def _design(self, games: pd.DataFrame, stints: pd.DataFrame):
        """Sparse matrix Z = [season dummies | player columns], rows = games."""
        row = pd.Series(np.arange(len(games)), index=games.game_id.to_numpy())
        st = stints[stints.game_id.isin(row.index)]
        r = row.loc[st.game_id].to_numpy()
        c = self._pcol.loc[st.player_id].to_numpy()
        vals = st.sign.to_numpy() * st.share.to_numpy()
        S = len(self._seasons)
        X = sp.csr_matrix((vals, (r, c + S)), shape=(len(games), S + len(self._players)))
        srow = np.arange(len(games))
        scol = pd.Series(np.arange(S), index=self._seasons).loc[games.season.to_numpy()].to_numpy()
        H = sp.csr_matrix((np.ones(len(games)), (srow, scol)), shape=X.shape)
        return (X + H).tocsc()

    def fit(self, games: pd.DataFrame, stints: pd.DataFrame, cutoff=None,
            prior: pd.Series | None = None) -> "PlusMinus":
        """Fit on games with line-ups dated before `cutoff` (no future data)."""
        g = games[games.has_lineups] if "has_lineups" in games else games
        if cutoff is not None:
            g = g[g.date < pd.Timestamp(cutoff)]
        g = g.reset_index(drop=True)
        if len(g) < 50:
            raise ValueError(f"Too few matches to fit ratings: {len(g)}")
        st = stints[stints.game_id.isin(g.game_id)]
        self._players = np.sort(st.player_id.unique())
        self._pcol = pd.Series(np.arange(len(self._players)), index=self._players)
        self._seasons = np.sort(g.season.unique())
        Z = self._design(g, st)
        y = (g.home_club_goals - g.away_club_goals).to_numpy(float)
        ref = pd.Timestamp(cutoff) if cutoff is not None else g.date.max()
        age = (ref - g.date).dt.days.to_numpy(float)
        w = 0.5 ** (age / self.halflife_days)
        mu = np.zeros(len(self._players)) if prior is None else \
            prior.reindex(self._players).fillna(0).to_numpy(float)
        S = len(self._seasons)
        d = np.r_[np.zeros(S), np.full(len(self._players), self.lam)]
        A = (Z.T @ sp.diags(w) @ Z + sp.diags(d)).tocsc()
        b = Z.T @ (w * y) + np.r_[np.zeros(S), self.lam * mu]
        self._lu = _Solver(A)
        self._Z, self._w, self._y, self._mu = Z, w, y, mu       # kept so we can draw posterior samples
        coef = self._lu.solve(b)
        resid = y - Z @ coef
        self.sigma2 = float(np.sum(w * resid ** 2) / np.sum(w))
        self.home = pd.Series(coef[:S], index=self._seasons, name="home_advantage")
        self.ratings = pd.Series(coef[S:], index=self._players, name="rating")
        self.prior = pd.Series(mu, index=self._players, name="prior")
        mins = st.groupby("player_id").share.sum() * 90
        self.minutes = mins.reindex(self._players).fillna(0).rename("minutes")
        self.cutoff, self.n_games = cutoff, len(g)
        return self

    def covariance(self, players) -> pd.DataFrame:
        """Posterior covariance block for a few players (one sparse solve per player)."""
        players = [p for p in players if p in self._pcol.index]
        S = len(self._seasons)
        cols = []
        for p in players:
            e = np.zeros(S + len(self._players)); e[S + self._pcol[p]] = 1.0
            cols.append(self._lu.solve(e))
        idx = [S + self._pcol[p] for p in players]
        block = self.sigma2 * np.array([[c[i] for i in idx] for c in cols])
        return pd.DataFrame(block, index=players, columns=players)

    def sd(self, players) -> pd.Series:
        c = self.covariance(players)
        return pd.Series(np.sqrt(np.diag(c.to_numpy())), index=c.index, name="sd")

    def sample(self, players, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw n joint posterior samples of the given players' ratings."""
        c = self.covariance(players)
        mean = self.ratings.loc[c.index].to_numpy()
        draws = rng.multivariate_normal(mean, c.to_numpy(), size=n, method="eigh")
        return pd.DataFrame(draws, columns=c.index)

    def sample_all(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw n JOINT posterior samples of every player's rating (rows = draws).

        Randomise-then-optimise: perturb the data y_m by N(0, sigma^2 / w_m) and the prior mean
        mu_i by N(0, sigma^2 / lam), then solve the same ridge system. The solution is an exact
        draw from N(beta_hat, sigma^2 A^-1) because
            Cov = A^-1 (sigma^2 Z'WZ + sigma^2 lam D) A^-1 = sigma^2 A^-1,   A = Z'WZ + lam D.
        One linear solve per draw, so it scales to every player at once, with all the
        correlations between team-mates included.
        """
        S, P = len(self._seasons), len(self._players)
        sd_y = np.sqrt(self.sigma2 / self._w)
        sd_mu = np.sqrt(self.sigma2 / self.lam)
        Zt = self._Z.T.tocsr()
        out = np.empty((n, P))
        for i in range(n):
            y_t = self._y + rng.standard_normal(len(self._y)) * sd_y
            mu_t = self._mu + rng.standard_normal(P) * sd_mu
            b = Zt @ (self._w * y_t) + np.r_[np.zeros(S), self.lam * mu_t]
            out[i] = self._lu.solve(b)[S:]
        return pd.DataFrame(out, columns=self._players)

    def rating_of(self, player, default: float | None = None) -> float:
        """Rating for any player; unseen players fall back to `default` (their prior)."""
        if player in self.ratings.index:
            return float(self.ratings[player])
        return 0.0 if default is None else float(default)

    def predict_gd(self, games: pd.DataFrame, stints: pd.DataFrame) -> np.ndarray:
        """Expected goal difference given line-ups (unknown players count as 0)."""
        st = stints[stints.game_id.isin(games.game_id)]
        contrib = st.sign * st.share * st.player_id.map(self.ratings).fillna(0)
        gd = contrib.groupby(st.game_id).sum().reindex(games.game_id).fillna(0).to_numpy()
        h = games.season.map(self.home).fillna(self.home.iloc[-1]).to_numpy(float)
        return h + gd


def tune(games: pd.DataFrame, stints: pd.DataFrame, cutoff, lams=(5, 10, 20, 40, 80),
         halflives=(365, 730, 1460), prior=None, valid_days: int = 365) -> pd.DataFrame:
    """Pick lam and halflife by fitting before (cutoff - valid_days) and scoring the next year."""
    cut = pd.Timestamp(cutoff)
    start = cut - pd.Timedelta(days=valid_days)
    valid = games[(games.date >= start) & (games.date < cut) & games.has_lineups]
    rows = []
    for lam in lams:
        for hl in halflives:
            m = PlusMinus(lam=lam, halflife_days=hl).fit(games, stints, start, prior)
            err = (valid.home_club_goals - valid.away_club_goals) - m.predict_gd(valid, stints)
            rows.append({"lam": lam, "halflife_days": hl, "mse": float(np.mean(err ** 2))})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
