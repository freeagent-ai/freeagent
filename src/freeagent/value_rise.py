"""Option B - Will a player's market value RISE over the next 12 months?

Unit: one row per player per season Y, at the cutoff T = 1 August of Y, for players who played at least
900 top-5-league minutes in season Y-1.
    label = 1  if  the latest market value in (T + 180 days, T + 365 days]  >  the latest value before T
Rows without a value before T (within the previous 365 days) or without a value in the label window are dropped.
Every feature uses information dated before T only (same rules as M8).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config

MIN_PREV_MINUTES = 900
NUMERIC = ["age", "age_sq", "height", "log_value", "value_trend", "prev_share", "prev2_share", "prev_ga90",
           "prev_euro_minutes", "club_ppg_prev", "moved_summer", "rating", "rating_minutes"]
CATEGORICAL = ["position", "foot", "to_league"]      # 'to_league' = the league of his club last season


def _asof(vals: pd.DataFrame, player_ids, when, strictly_before: bool = True):
    """Latest valuation (value, date) for each player strictly before `when` (a Series of dates)."""
    q = pd.DataFrame({"player_id": np.asarray(player_ids), "when": pd.to_datetime(np.asarray(when)),
                      "i": np.arange(len(player_ids))}).sort_values("when")
    v = vals[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    r = pd.merge_asof(q, v, left_on="when", right_on="date", by="player_id", allow_exact_matches=not strictly_before)
    r = r.sort_values("i")
    return r.market_value_in_eur.to_numpy(), r.date.to_numpy()


def build_panel(tables: dict, lt: dict, seasons, ratings: dict | None = None) -> pd.DataFrame:
    vals = tables.get("player_valuations")
    if vals is None or len(vals) == 0:
        raise ValueError("player_valuations table is required")
    vals = vals.dropna(subset=["date", "market_value_in_eur"])
    vals = vals[vals.market_value_in_eur > 0]
    pcs = lt["pcs"]
    tot = pcs.groupby(["player_id", "season"]).agg(mins=("mins", "sum"), g=("ga", "sum"), a=("ast", "sum"),
                                                   n_games=("n_games", "max")).reset_index()
    tot["share"] = (tot.mins / (90 * tot.n_games)).clip(upper=1)
    tot["ga90"] = np.where(tot.mins >= 450, 90 * (tot.g + tot.a) / tot.mins.clip(lower=1), np.nan)
    key = tot.set_index(["player_id", "season"])
    main_club = pcs.sort_values("mins").groupby(["player_id", "season"]).tail(1).set_index(["player_id", "season"])
    club = lt["club"].set_index(["club_id", "season"])
    euro = lt["euro"].set_index(["player_id", "season"]).euro_minutes
    players = tables["players"].set_index("player_id")
    col = lambda c: players[c] if c in players else pd.Series(dtype=object)
    tr = tables["transfers"].dropna(subset=["transfer_date"])
    rows = []
    for Y in seasons:
        T = pd.Timestamp(f"{Y}-08-01")
        prev = tot[(tot.season == Y - 1) & (tot.mins >= MIN_PREV_MINUTES)]
        if prev.empty:
            continue
        s = pd.DataFrame({"player_id": prev.player_id.to_numpy(), "season": Y})
        when = np.full(len(s), T)
        v_now, d_now = _asof(vals, s.player_id, when)
        v_1y, _ = _asof(vals, s.player_id, np.full(len(s), T - pd.Timedelta(days=365)))
        v_fut, d_fut = _asof(vals, s.player_id, np.full(len(s), T + pd.Timedelta(days=365)), strictly_before=False)
        s["value_now"], s["value_future"] = v_now, v_fut
        s["value_now_date"], s["value_future_date"] = d_now, d_fut
        i1 = pd.MultiIndex.from_arrays([s.player_id, np.full(len(s), Y - 1)])
        i2 = pd.MultiIndex.from_arrays([s.player_id, np.full(len(s), Y - 2)])
        s["prev_share"] = key.share.reindex(i1).to_numpy()
        s["prev2_share"] = key.share.reindex(i2).to_numpy()
        s["prev_ga90"] = key.ga90.reindex(i1).to_numpy()
        s["prev_euro_minutes"] = euro.reindex(i1).fillna(0).to_numpy()
        mc = main_club.player_club_id.reindex(i1).to_numpy()
        ci = pd.MultiIndex.from_arrays([mc, np.full(len(s), Y - 1)])
        s["club_ppg_prev"] = club.ppg.reindex(ci).to_numpy()
        s["to_league"] = club.league.reindex(ci).to_numpy()
        moved = tr[(tr.transfer_date >= pd.Timestamp(f"{Y}-06-01")) & (tr.transfer_date < T)].player_id.unique()
        s["moved_summer"] = s.player_id.isin(moved).astype(int)
        s["age"] = (T - s.player_id.map(players.date_of_birth)).dt.days / 365.25
        s["position"] = s.player_id.map(col("position")).fillna("Unknown")
        s["foot"] = s.player_id.map(col("foot")).fillna("Unknown")
        s["height"] = pd.to_numeric(s.player_id.map(col("height_in_cm")), errors="coerce")
        s["value_trend"] = v_now / np.where(v_1y > 0, v_1y, np.nan)
        s["rating"], s["rating_minutes"] = np.nan, np.nan
        if ratings and Y in ratings:
            s["rating"] = s.player_id.map(ratings[Y].rating).to_numpy()
            s["rating_minutes"] = s.player_id.map(ratings[Y].minutes).to_numpy()
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    now_ok = pd.notna(df.value_now) & (pd.to_datetime(df.value_now_date) >= df.season.map(
        lambda y: pd.Timestamp(f"{y}-08-01") - pd.Timedelta(days=365)))
    fut_ok = pd.notna(df.value_future) & (pd.to_datetime(df.value_future_date) > df.season.map(
        lambda y: pd.Timestamp(f"{y}-08-01") + pd.Timedelta(days=180)))
    df = df[now_ok & fut_ok].reset_index(drop=True)
    df["log_value"] = np.log10(df.value_now)
    df["age_sq"] = df.age ** 2
    df["rise"] = (df.value_future > df.value_now).astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def walk_forward(df: pd.DataFrame, model_fn, features, test_seasons, label: str = "rise", min_season: int = 2015):
    out = []
    for y in test_seasons:
        tr, te = df[(df.season < y) & (df.season >= min_season)], df[df.season == y]
        if len(te) == 0 or len(tr) == 0:
            continue
        m = model_fn().fit(tr[features], tr[label])
        out.append(te.assign(p=m.predict_proba(te[features])[:, 1]))
    return pd.concat(out)
