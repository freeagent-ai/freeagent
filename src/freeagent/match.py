"""M4 - Match outcome models.

All models output P(home win), P(draw), P(away win) for a match.

Goals are Poisson. For a team in a match:  log(lambda) = linear predictor.
We fit by maximum likelihood with our own analytic gradient (no black box):
    loglik = sum_rows w * (g * log(lambda) - lambda)        (constant log g! dropped)
    gradient wrt coefficients = Z' [ w * (g - lambda) ]
Then Dixon-Coles adds rho, a correction for the low scores 0-0, 1-0, 0-1, 1-1,
fitted in a second step by one-dimensional maximum likelihood.

Models
- TeamDixonColes  : attack/defence per team (classic Dixon & Coles 1997).          Baseline
- PlayerPoisson   : strength from player ratings of the EXPECTED line-up.          Ours
                    log l_home = c + h + k*(S_home - S_away);  log l_away = c - k*(S_home - S_away)
- Elo, Frequency  : simple baselines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


# ---------- shared maths ----------
def tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction factor."""
    t = np.ones(np.broadcast(x, y, lh).shape)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def outcome_probs(lh: np.ndarray, la: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """Vectorised P(home, draw, away) for arrays of goal rates. Shape (n, 3)."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lh[:, None])                 # (n, G)
    pa = poisson.pmf(g[None, :], la[:, None])
    m = ph[:, :, None] * pa[:, None, :]                       # (n, G, G)
    if rho:
        corr = tau(g[:, None], g[None, :], lh[:, None, None], la[:, None, None], rho)
        m = m * corr
    m = m / m.sum(axis=(1, 2), keepdims=True)
    home = np.tril(np.ones((len(g), len(g))), -1)             # rows = home goals
    return np.stack([(m * home).sum((1, 2)), np.trace(m, axis1=1, axis2=2), (m * home.T).sum((1, 2))], 1)


def score_matrix(lh: float, la: float, rho: float = 0.0) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)) * tau(g[:, None], g[None, :], lh, la, rho)
    return m / m.sum()


def fit_poisson_glm(Z, goals, w, l2: np.ndarray) -> np.ndarray:
    """Weighted Poisson regression by L-BFGS with analytic gradient; l2 = per-coef penalty."""
    def f(b):
        eta = Z @ b
        lam = np.exp(eta)
        nll = -np.sum(w * (goals * eta - lam)) + 0.5 * np.sum(l2 * b * b)
        grad = -(Z.T @ (w * (goals - lam))) + l2 * b
        return nll, grad
    res = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Poisson fit failed: {res.message}")
    return res.x


def fit_rho(hg, ag, lh, la, w) -> float:
    """Second-step MLE of the Dixon-Coles rho given fitted goal rates."""
    def nll(rho):
        t = tau(hg, ag, lh, la, rho)
        if np.any(t <= 0):
            return 1e12
        return -np.sum(w * np.log(t))
    return float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)


def decay_weights(dates: pd.Series, ref, halflife_days: float) -> np.ndarray:
    age = (pd.Timestamp(ref) - dates).dt.days.to_numpy(float)
    return 0.5 ** (age / halflife_days)


def _train(games, cutoff):
    g = games[games.date < pd.Timestamp(cutoff)] if cutoff is not None else games
    if len(g) < 50:
        raise ValueError("Too few training matches")
    return g.reset_index(drop=True)


# ---------- models ----------
class TeamDixonColes:
    """Classic team-strength model: log l = mu + home + attack[team] + defence[opponent]."""
    def __init__(self, halflife_days: float = 365.0, penalty: float = 1.0):
        self.halflife_days, self.penalty = halflife_days, penalty

    def fit(self, games: pd.DataFrame, cutoff=None) -> "TeamDixonColes":
        g = _train(games, cutoff)
        teams = np.unique(np.r_[g.home_club_id, g.away_club_id])
        self.idx = pd.Series(np.arange(len(teams)), index=teams)
        n, T = len(g), len(teams)
        h, a = self.idx.loc[g.home_club_id].to_numpy(), self.idx.loc[g.away_club_id].to_numpy()
        rows = np.arange(2 * n)
        team = np.r_[h, a]; opp = np.r_[a, h]
        Z = sp.csr_matrix((np.ones(2 * n), (rows, np.zeros(2 * n, int))), shape=(2 * n, 2 + 2 * T))
        Z = Z + sp.csr_matrix((np.r_[np.ones(n), np.zeros(n)], (rows, np.ones(2 * n, int))), shape=Z.shape)
        Z = Z + sp.csr_matrix((np.ones(2 * n), (rows, 2 + team)), shape=Z.shape)
        Z = Z + sp.csr_matrix((np.ones(2 * n), (rows, 2 + T + opp)), shape=Z.shape)
        goals = np.r_[g.home_club_goals, g.away_club_goals].astype(float)
        ref = cutoff if cutoff is not None else g.date.max()
        w1 = decay_weights(g.date, ref, self.halflife_days); w = np.r_[w1, w1]
        l2 = np.r_[0, 0, np.full(2 * T, self.penalty)]
        self.coef = fit_poisson_glm(Z.tocsr(), goals, w, l2)
        lam = np.exp(Z @ self.coef)
        self.rho = fit_rho(g.home_club_goals.to_numpy(), g.away_club_goals.to_numpy(), lam[:n], lam[n:], w1)
        self.T = T
        return self

    def rates(self, games: pd.DataFrame):
        mu, home = self.coef[0], self.coef[1]
        att, dfn = self.coef[2:2 + self.T], self.coef[2 + self.T:]
        def get(arr, ids):  # unseen (e.g. promoted) teams get the average (0)
            ix = self.idx.reindex(ids).to_numpy()
            return np.array([arr[int(i)] if not np.isnan(i) else 0.0 for i in ix])
        h, a = games.home_club_id.to_numpy(), games.away_club_id.to_numpy()
        lh = np.exp(mu + home + get(att, h) + get(dfn, a))
        la = np.exp(mu + get(att, a) + get(dfn, h))
        return lh, la

    def predict(self, games: pd.DataFrame) -> np.ndarray:
        lh, la = self.rates(games)
        return outcome_probs(lh, la, self.rho)


class PlayerPoisson:
    """Our model: team strength = rating of the expected line-up (needs S_home, S_away columns)."""
    def __init__(self, halflife_days: float = 365.0):
        self.halflife_days = halflife_days

    def fit(self, games: pd.DataFrame, cutoff=None) -> "PlayerPoisson":
        g = _train(games.dropna(subset=["S_home", "S_away"]), cutoff)
        n = len(g); gap = (g.S_home - g.S_away).to_numpy()
        Z = np.column_stack([np.ones(2 * n), np.r_[np.ones(n), np.zeros(n)], np.r_[gap, -gap]])
        goals = np.r_[g.home_club_goals, g.away_club_goals].astype(float)
        ref = cutoff if cutoff is not None else g.date.max()
        w1 = decay_weights(g.date, ref, self.halflife_days); w = np.r_[w1, w1]
        self.coef = fit_poisson_glm(sp.csr_matrix(Z), goals, w, np.zeros(3))
        lam = np.exp(Z @ self.coef)
        self.rho = fit_rho(g.home_club_goals.to_numpy(), g.away_club_goals.to_numpy(), lam[:n], lam[n:], w1)
        return self

    def rates(self, games: pd.DataFrame, shift_home: float = 0.0, shift_away: float = 0.0):
        c, h, k = self.coef
        gap = (games.S_home.fillna(0) + shift_home - games.S_away.fillna(0) - shift_away).to_numpy()
        return np.exp(c + h + k * gap), np.exp(c - k * gap)

    def predict(self, games: pd.DataFrame) -> np.ndarray:
        lh, la = self.rates(games)
        return outcome_probs(lh, la, self.rho)


class HybridPoisson(PlayerPoisson):
    """Our final model: player ratings (who is in the squad) + Elo gap (current form).
        log l_home = c + h + k*dS + g*dE ;  log l_away = c - k*dS - g*dE
    dS = expected line-up strength gap, dE = pre-match Elo gap / 400.
    Needs columns S_home, S_away, elo_gap."""
    def fit(self, games: pd.DataFrame, cutoff=None) -> "HybridPoisson":
        g = _train(games.dropna(subset=["S_home", "S_away", "elo_gap"]), cutoff)
        n = len(g)
        ds, de = (g.S_home - g.S_away).to_numpy(), g.elo_gap.to_numpy()
        Z = np.column_stack([np.ones(2 * n), np.r_[np.ones(n), np.zeros(n)], np.r_[ds, -ds], np.r_[de, -de]])
        goals = np.r_[g.home_club_goals, g.away_club_goals].astype(float)
        ref = cutoff if cutoff is not None else g.date.max()
        w1 = decay_weights(g.date, ref, self.halflife_days); w = np.r_[w1, w1]
        self.coef = fit_poisson_glm(sp.csr_matrix(Z), goals, w, np.zeros(4))
        lam = np.exp(Z @ self.coef)
        self.rho = fit_rho(g.home_club_goals.to_numpy(), g.away_club_goals.to_numpy(), lam[:n], lam[n:], w1)
        return self

    def rates(self, games: pd.DataFrame, shift_home: float = 0.0, shift_away: float = 0.0):
        c, h, k, ge = self.coef
        ds = (games.S_home.fillna(0) + shift_home - games.S_away.fillna(0) - shift_away).to_numpy()
        de = games.elo_gap.fillna(0).to_numpy()
        return np.exp(c + h + k * ds + ge * de), np.exp(c - k * ds - ge * de)


def elo_gaps(games: pd.DataFrame, k: float = 20.0, home_bonus: float = 60.0) -> pd.DataFrame:
    """Pre-match Elo gap (home minus away, divided by 400) for every game, in date order."""
    g = games.sort_values(["date", "game_id"])
    r, out = {}, []
    for row in g.itertuples():
        rh, ra = r.get(row.home_club_id, 1500.0), r.get(row.away_club_id, 1500.0)
        out.append((rh - ra) / 400)
        e = 1 / (1 + 10 ** (-(rh + home_bonus - ra) / 400))
        s = 1.0 if row.home_club_goals > row.away_club_goals else 0.5 if row.home_club_goals == row.away_club_goals else 0.0
        r[row.home_club_id], r[row.away_club_id] = rh + k * (s - e), ra - k * (s - e)
    return pd.DataFrame({"game_id": g.game_id.to_numpy(), "elo_gap": out})


class Elo:
    """Elo ratings (K, home bonus) -> probabilities via a multinomial logit on the Elo gap."""
    def __init__(self, k: float = 20.0, home_bonus: float = 60.0):
        self.k, self.home_bonus = k, home_bonus

    def _run(self, games):
        r, gaps = {}, []
        for row in games.itertuples():
            rh, ra = r.get(row.home_club_id, 1500.0), r.get(row.away_club_id, 1500.0)
            gaps.append(rh + self.home_bonus - ra)
            e = 1 / (1 + 10 ** (-(gaps[-1]) / 400))
            s = 1.0 if row.home_club_goals > row.away_club_goals else 0.5 if row.home_club_goals == row.away_club_goals else 0.0
            r[row.home_club_id], r[row.away_club_id] = rh + self.k * (s - e), ra - self.k * (s - e)
        return np.array(gaps)

    def fit(self, games: pd.DataFrame, cutoff=None) -> "Elo":
        from sklearn.linear_model import LogisticRegression
        self._all = games.sort_values(["date", "game_id"]).reset_index(drop=True)
        self._gap = pd.Series(self._run(self._all), index=self._all.game_id)   # pre-match, no leakage
        g = _train(self._all, cutoff)
        self.clf = LogisticRegression(max_iter=1000).fit(self._gap.loc[g.game_id].to_numpy()[:, None] / 400,
                                                         outcome_label(g))
        return self

    def predict(self, games: pd.DataFrame) -> np.ndarray:
        x = self._gap.reindex(games.game_id).fillna(self.home_bonus).to_numpy()[:, None] / 400
        p = self.clf.predict_proba(x)
        return p[:, np.argsort(self.clf.classes_)]


class Frequency:
    """Always predicts the training share of home wins / draws / away wins."""
    def fit(self, games, cutoff=None):
        y = outcome_label(_train(games, cutoff))
        self.p = (np.bincount(y, minlength=3) + 1) / (len(y) + 3)
        return self

    def predict(self, games):
        return np.tile(self.p, (len(games), 1))


# ---------- expected line-up strength (pre-match information only) ----------
def expected_strength(games: pd.DataFrame, stints: pd.DataFrame, ratings: pd.Series,
                      fallback: pd.Series | None = None, window: int = 5) -> pd.DataFrame:
    """For each game, each side's strength = sum(rating * average minutes share over the
    club's previous `window` matches). Uses only matches BEFORE the game, never its own line-up."""
    rate = ratings if fallback is None else ratings.combine_first(fallback)
    order = games.set_index("game_id").date
    out = {}
    st = stints.assign(date=stints.game_id.map(order)).dropna(subset=["date"])
    for club, s in st.groupby("club_id"):
        wide = s.pivot_table(index=["date", "game_id"], columns="player_id", values="share", aggfunc="sum").fillna(0)
        wide = wide.sort_index()
        hist = wide.rolling(window, min_periods=1).mean().shift(1)            # previous matches only
        strength = hist.to_numpy() @ rate.reindex(wide.columns).fillna(0).to_numpy()
        for (d, gid), val in zip(wide.index, strength):
            out[(gid, club)] = val
    S = pd.Series(out)
    g = games[["game_id", "home_club_id", "away_club_id"]]
    sh = [S.get((a, b), np.nan) for a, b in zip(g.game_id, g.home_club_id)]
    sa = [S.get((a, b), np.nan) for a, b in zip(g.game_id, g.away_club_id)]
    return pd.DataFrame({"game_id": g.game_id.to_numpy(), "S_home": sh, "S_away": sa})


def honest_strength(games: pd.DataFrame, stints: pd.DataFrame, seasons, rating_fn) -> pd.DataFrame:
    """Cross-fitted strengths: games in season s get strengths from ratings fitted only on data
    before 1 August of s. Without this, training-period strengths are in-sample and the match
    model over-trusts them (we measured this). rating_fn(cutoff) -> pd.Series of ratings."""
    parts = []
    for s in seasons:
        cut, end = pd.Timestamp(f"{s}-08-01"), pd.Timestamp(f"{s + 1}-07-31")
        part = games[(games.date >= cut) & (games.date <= end)]
        if part.empty:
            continue
        try:
            ratings = rating_fn(cut)
        except ValueError:          # not enough history before this season: skip it
            continue
        st = expected_strength(games[games.date <= end], stints, ratings)
        parts.append(st[st.game_id.isin(part.game_id)])
    return pd.concat(parts, ignore_index=True)


# ---------- metrics ----------
def outcome_label(games: pd.DataFrame) -> np.ndarray:
    return np.where(games.home_club_goals > games.away_club_goals, 0,
                    np.where(games.home_club_goals == games.away_club_goals, 1, 2))


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1); p = p / p.sum(1, keepdims=True)
    onehot = np.eye(3)[y]
    cum_p, cum_y = np.cumsum(p, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    return {"log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
            "rps": float(np.mean(np.sum((cum_p - cum_y) ** 2, 1) / 2)),
            "brier": float(np.mean(np.sum((p - onehot) ** 2, 1))),
            "accuracy": float(np.mean(p.argmax(1) == y)),
            "n": int(len(y))}
