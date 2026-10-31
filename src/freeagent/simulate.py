"""M5 - Monte Carlo season simulator: how many league points does one signing add?

Ingredients
  * strength of a club S = sum_i rating_i * (expected minutes share_i)       (M3 ratings)
  * match model (M4 PlayerPoisson):  home goals ~ Poisson(exp(c + h + k (S_h - S_a))),
                                     away goals ~ Poisson(exp(c     - k (S_h - S_a)))
  * the signing changes his club's strength by
        Delta = s_new * (r_new - r_inc)
    s_new = minutes share he will get (0..1), r_new = his rating, r_inc = minutes-weighted
    average rating of the same-position players he displaces. They lose exactly the minutes
    he gains, so only the difference in quality counts ("displacement-aware").

Two Monte Carlo layers
  A. Uncertainty about Delta (what we do not know when we sign him):
        r_new ~ Normal(rating, posterior sd)           from M3's Bayesian posterior
        regular ~ Bernoulli(p)                         from the M8 classifier
        s_new   ~ drawn from real first-season minutes shares of past signings
     -> thousands of Delta draws.
  B. Match randomness: the whole league season is played many times with Poisson goals,
     with and without the signing, on the SAME random numbers (common random numbers),
     so the difference in points is clean of luck.

Because expected points are a smooth function of Delta, layer A uses an exact curve
f(Delta) = E[points of the club | Delta] evaluated on a grid and interpolated, instead of
simulating every draw. Layer B is used to turn Delta into title / top-4 / relegation odds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .match import outcome_probs

WINDOW = 5      # a club's baseline line-up = its last 5 matches before the season


# --------------------------------------------------------------------------- baseline squad
def baseline_squad(games: pd.DataFrame, stints: pd.DataFrame, ratings: pd.Series,
                   players: pd.DataFrame, season: int, window: int = WINDOW) -> pd.DataFrame:
    """Each club's expected line-up at the start of `season`: for every player, his average
    minutes share over the club's last `window` matches before 1 July of `season`.
    Returns club_id, player_id, share, rating, position."""
    cut = pd.Timestamp(f"{season}-07-01")
    g = games[(games.date < cut) & games.has_lineups]
    long = pd.concat([g[["game_id", "date", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                      g[["game_id", "date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    long = long.sort_values(["club_id", "date", "game_id"])
    last = long.groupby("club_id").tail(window)
    st = stints.merge(last[["game_id", "club_id"]], on=["game_id", "club_id"])
    n = last.groupby("club_id").size()
    sq = st.groupby(["club_id", "player_id"]).share.sum().reset_index()
    sq["share"] = sq.share / sq.club_id.map(n)
    sq["rating"] = sq.player_id.map(ratings).fillna(0.0)
    sq["position"] = sq.player_id.map(players.set_index("player_id").position).fillna("Unknown")
    return sq


def season_squad(games: pd.DataFrame, stints: pd.DataFrame, ratings: pd.Series, players: pd.DataFrame,
                 transfers: pd.DataFrame, season: int, new_share: float, fallback: pd.Series | None = None,
                 window: int = WINDOW) -> pd.DataFrame:
    """Pre-season squad that ALSO knows the summer window up to 1 August of `season`:
    players who left (transfer from the club, 1 June - 31 July) are removed, players who arrived are added
    with minutes share `new_share`, then every club's shares are rescaled to its baseline total (a team still
    plays 11 players). Arrivals without a rating get `fallback` (the M2 prior) or 0 (league average)."""
    sq = baseline_squad(games, stints, ratings, players, season, window)
    start, cut = pd.Timestamp(f"{season}-06-01"), pd.Timestamp(f"{season}-08-01")
    tr = transfers.dropna(subset=["transfer_date"])
    tr = tr[(tr.transfer_date >= start) & (tr.transfer_date < cut)].sort_values("transfer_date")
    total = sq.groupby("club_id").share.sum()
    left = pd.MultiIndex.from_frame(tr[["from_club_id", "player_id"]].rename(columns={"from_club_id": "club_id"}))
    key = pd.MultiIndex.from_frame(sq[["club_id", "player_id"]])
    sq = sq[~key.isin(left)]
    arr = (tr[tr.to_club_id.isin(total.index)][["to_club_id", "player_id"]]
           .rename(columns={"to_club_id": "club_id"}).drop_duplicates())
    arr = arr[~pd.MultiIndex.from_frame(arr).isin(pd.MultiIndex.from_frame(sq[["club_id", "player_id"]]))]
    # a player who moved on again before 1 August is not in the squad
    last_move = tr.groupby("player_id").tail(1)
    arr = arr.merge(last_move[["player_id", "to_club_id"]], on="player_id")
    arr = arr[arr.club_id == arr.to_club_id].drop(columns="to_club_id")
    r = arr.player_id.map(ratings)
    if fallback is not None:
        r = r.fillna(arr.player_id.map(fallback))
    arr = arr.assign(share=new_share, rating=r.fillna(0.0).to_numpy(),
                     position=arr.player_id.map(players.set_index("player_id").position).fillna("Unknown"))
    out = pd.concat([sq, arr], ignore_index=True)
    scale = out.club_id.map(total) / out.club_id.map(out.groupby("club_id").share.sum())
    out["share"] = out.share * scale.to_numpy()
    return out


def promoted_strength(S: pd.Series, clubs_now, clubs_last) -> pd.Series:
    """Promoted clubs (in the league now, not last season) get the average strength of the clubs they replaced
    (in the league last season, not now). Our data has no second-division matches, so without this a promoted
    club has no line-up (strength 0 = league average) or a line-up from years ago."""
    clubs_now, clubs_last = pd.Index(clubs_now), pd.Index(clubs_last)
    up, down = clubs_now.difference(clubs_last), clubs_last.difference(clubs_now)
    known = [c for c in down if c in S.index]
    if len(up) == 0 or not known:
        return S
    S = S.copy()
    for c in up:
        S[c] = float(S[known].mean())
    return S


def club_strength(squad: pd.DataFrame) -> pd.Series:
    return (squad.share * squad.rating).groupby(squad.club_id).sum().rename("S")


def displaced_rating(squad: pd.DataFrame, club_id: int, position: str, exclude: int | None = None) -> float:
    """Minutes-weighted average rating of the club's players in `position` (they lose his minutes
    in proportion to how much they play now). Falls back to the whole squad if none."""
    s = squad[(squad.club_id == club_id) & (squad.player_id != exclude)]
    pos = s[s.position == position]
    use = pos if len(pos) and pos.share.sum() > 0 else s
    if len(use) == 0 or use.share.sum() <= 0:
        return 0.0
    return float(np.average(use.rating, weights=use.share))


def without_player(squad: pd.DataFrame, club_id: int, player_id: int) -> pd.DataFrame:
    """The club's squad as it would be WITHOUT this signing: his row removed and the other players' shares
    rescaled to the club's total (exactly what season_squad gives if he had not arrived)."""
    s = squad[squad.club_id == club_id]
    rest = s[s.player_id != player_id].copy()
    if len(rest) and rest.share.sum() > 0:
        rest["share"] = rest.share * s.share.sum() / rest.share.sum()
    return rest


def displacement_order(club_squad: pd.DataFrame, position: str):
    """Who loses minutes when a signing plays: a manager drops his WEAKEST players first. Order = players in
    the signing's position from the lowest rating up, then the rest of the squad from the lowest rating up.
    Returns (cumulative share, cumulative share x rating), both starting at 0."""
    same = club_squad[club_squad.position == position].sort_values("rating", kind="stable")
    rest = club_squad[club_squad.position != position].sort_values("rating", kind="stable")
    o = pd.concat([same, rest])
    return (np.r_[0.0, np.cumsum(o.share.to_numpy(float))],
            np.r_[0.0, np.cumsum((o.share * o.rating).to_numpy(float))])


def displaced_mass(order, share: np.ndarray) -> np.ndarray:
    """Sum of share x rating of the minutes a signing with minutes share `share` takes (weakest first).
    Minutes beyond the whole squad would go to a league-average (rating 0) player, so they add nothing."""
    cum, mass = order
    return np.interp(np.minimum(share, cum[-1]), cum, mass)


# --------------------------------------------------------------------------- layer A
def delta_draws_weakest(rating: float, sd: float, p_regular: float, order, pool_regular: np.ndarray,
                        pool_other: np.ndarray, n: int, rng: np.random.Generator):
    """Draw n values of Delta = s_new * r_new - (share x rating of the weakest minutes he takes).
    Returns (Delta draws, average rating of the minutes he displaces)."""
    is_reg = rng.random(n) < p_regular
    share = np.where(is_reg, rng.choice(pool_regular, n), rng.choice(pool_other, n))
    r_new = rng.normal(rating, max(sd, 1e-9), n)
    lost = displaced_mass(order, share)
    r_inc = float(lost.sum() / share.sum()) if share.sum() > 0 else 0.0
    return share * r_new - lost, r_inc


def delta_draws(rating: float, sd: float, p_regular: float, r_inc: float, pool_regular: np.ndarray,
                pool_other: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n values of Delta = s_new * (r_new - r_inc)."""
    is_reg = rng.random(n) < p_regular
    share = np.where(is_reg, rng.choice(pool_regular, n), rng.choice(pool_other, n))
    r_new = rng.normal(rating, max(sd, 1e-9), n)
    return share * (r_new - r_inc)


def _lambdas(model, S_home, S_away):
    c, h, k = model.coef
    gap = S_home - S_away
    return np.exp(c + h + k * gap), np.exp(c - k * gap)


def points_curve(model, fixtures: pd.DataFrame, S: pd.Series, club_id: int, grid: np.ndarray) -> np.ndarray:
    """Expected league points of `club_id` as a function of its strength shift Delta (exact)."""
    home = (fixtures.home_club_id == club_id).to_numpy()
    mine = fixtures[home | (fixtures.away_club_id == club_id).to_numpy()]
    home = (mine.home_club_id == club_id).to_numpy()
    Sh0 = mine.home_club_id.map(S).fillna(0).to_numpy()
    Sa0 = mine.away_club_id.map(S).fillna(0).to_numpy()
    out = []
    for d in grid:
        lh, la = _lambdas(model, Sh0 + d * home, Sa0 + d * ~home)
        P = outcome_probs(lh, la, model.rho)
        out.append(np.where(home, 3 * P[:, 0] + P[:, 1], 3 * P[:, 2] + P[:, 1]).sum())
    return np.array(out)


def points_gain(model, fixtures, S, club_id, deltas: np.ndarray, grid_size: int = 41) -> np.ndarray:
    """Extra expected points for each Delta draw (interpolated exact curve, minus the Delta = 0 value)."""
    lo, hi = min(deltas.min(), -1e-3), max(deltas.max(), 1e-3)
    grid = np.linspace(lo, hi, grid_size)
    curve = points_curve(model, fixtures, S, club_id, np.unique(np.r_[grid, 0.0]))
    g = np.unique(np.r_[grid, 0.0])
    return np.interp(deltas, g, curve) - np.interp(0.0, g, curve)


def arrival_share(transfers: pd.DataFrame, pcs: pd.DataFrame, before_season: int,
                  min_fee: float = config.MIN_FEE_EUR) -> float:
    """Minutes share given to every summer arrival in the pre-season squad: the average first-season league
    share of summer arrivals in EARLIER seasons (mean of the big-fee and the other group, as in season_predict)."""
    tr = transfers.dropna(subset=["transfer_date"])
    tr = tr[tr.transfer_date.dt.month.isin([6, 7]) & (tr.transfer_date.dt.year < before_season)].copy()
    tr["season"] = tr.transfer_date.dt.year
    m = tr.merge(pcs[["player_id", "player_club_id", "season", "share"]],
                 left_on=["player_id", "to_club_id", "season"], right_on=["player_id", "player_club_id", "season"])
    big = m.transfer_fee.fillna(0) >= min_fee
    sb = float(m.share[big].mean()) if big.any() else 0.4
    so = float(m.share[~big].mean()) if (~big).any() else 0.2
    return (sb + so) / 2


def signing_points(model, fixtures: pd.DataFrame, squad: pd.DataFrame, S: pd.Series, club_id: int,
                   player_id: int, position: str, rating: float, sd: float, p_regular: float,
                   pool_regular: np.ndarray, pool_other: np.ndarray, n: int, rng: np.random.Generator):
    """Extra league points one signing adds to his club (M5, displacement-aware, weakest players drop out).
    `squad` is the summer-aware pre-season squad (season_squad); the club's strength without him is
    recomputed so he is never counted twice. Returns (Delta draws, points-gain draws, displaced rating)."""
    rest = without_player(squad, club_id, player_id)
    S2 = S.copy()
    S2[club_id] = float((rest.share * rest.rating).sum())
    order = displacement_order(rest, position)
    dl, r_inc = delta_draws_weakest(rating, sd, p_regular, order, pool_regular, pool_other, n, rng)
    return dl, points_gain(model, fixtures, S2, club_id, dl), r_inc


# --------------------------------------------------------------------------- layer B
def simulate_table(model, fixtures: pd.DataFrame, S: pd.Series, shifts: dict | None = None,
                   n_sims: int = 2000, seed: int = config.SEED) -> pd.DataFrame:
    """Play the whole league season n_sims times. `shifts` = {club_id: Delta}.
    Same seed => same random numbers, so with/without runs are directly comparable."""
    S2 = S.copy()
    for c, d in (shifts or {}).items():
        S2[c] = S2.get(c, 0.0) + d
    rng = np.random.default_rng(seed)
    clubs = np.sort(pd.unique(np.r_[fixtures.home_club_id, fixtures.away_club_id]))
    idx = pd.Series(np.arange(len(clubs)), index=clubs)
    h, a = idx[fixtures.home_club_id].to_numpy(), idx[fixtures.away_club_id].to_numpy()
    lh, la = _lambdas(model, S2.reindex(fixtures.home_club_id).fillna(0).to_numpy(),
                      S2.reindex(fixtures.away_club_id).fillna(0).to_numpy())
    u1, u2 = rng.random((2, len(fixtures), n_sims))
    from scipy.stats import poisson
    gh, ga = poisson.ppf(u1, lh[:, None]), poisson.ppf(u2, la[:, None])
    pts_h = np.where(gh > ga, 3, np.where(gh == ga, 1, 0))
    pts_a = np.where(ga > gh, 3, np.where(gh == ga, 1, 0))
    pts = np.zeros((len(clubs), n_sims))
    np.add.at(pts, h, pts_h)
    np.add.at(pts, a, pts_a)
    gd = np.zeros((len(clubs), n_sims))
    np.add.at(gd, h, gh - ga)
    np.add.at(gd, a, ga - gh)
    rank = (-(pts + 1e-3 * gd)).argsort(0).argsort(0) + 1       # 1 = champion, ties broken by goal difference
    n = len(clubs)
    return pd.DataFrame({"mean_points": pts.mean(1), "points_p10": np.percentile(pts, 10, axis=1),
                         "points_p90": np.percentile(pts, 90, axis=1), "p_title": (rank == 1).mean(1),
                         "p_top4": (rank <= 4).mean(1), "p_top_half": (rank <= n // 2).mean(1),
                         "p_relegation": (rank > n - 3).mean(1),
                         "mean_rank": rank.mean(1)}, index=pd.Index(clubs, name="club_id"))
