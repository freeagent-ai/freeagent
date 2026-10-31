"""M1 - Data pipeline.

Input : the Kaggle Transfermarkt CSVs (a folder or the freeagent_subset.zip).
Output: clean tables plus a long "stint" table: one row per (match, player) with
        sign (+1 home side, -1 away side) and share = minutes played / 90.
        That stint table is exactly the design matrix the plus-minus model needs.
"""
from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from . import config

TABLES = ["games", "appearances", "transfers", "players", "clubs", "competitions",
          "player_valuations", "game_lineups", "game_events"]
DATE_COLUMNS = {"games": ["date"], "appearances": ["date"], "transfers": ["transfer_date"],
                "players": ["date_of_birth"], "player_valuations": ["date"]}


def load_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read every known table from a folder or zip. Missing optional tables are skipped."""
    path = Path(path)
    out = {}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = {Path(n).stem.split(".")[0]: n for n in z.namelist()}
            for t in TABLES:
                if t in names:
                    with z.open(names[t]) as f:
                        out[t] = pd.read_csv(f, low_memory=False)
    else:
        for t in TABLES:
            for ext in (".csv", ".csv.gz"):
                if (path / f"{t}{ext}").exists():
                    out[t] = pd.read_csv(path / f"{t}{ext}", low_memory=False)
                    break
    for t in ["games", "appearances", "transfers", "players", "clubs"]:
        if t not in out:
            raise FileNotFoundError(f"Required table '{t}' not found in {path}")
    return standardise(out)


def standardise(t: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Consistent dtypes: dates as Timestamps, ids as int, fees as float."""
    t = {k: v.copy() for k, v in t.items()}
    for table, cols in DATE_COLUMNS.items():
        if table in t:
            for c in cols:
                if c in t[table]:
                    t[table][c] = pd.to_datetime(t[table][c], errors="coerce")
    g = t["games"]
    g["season"] = pd.to_numeric(g["season"], errors="coerce").astype("Int64")
    if "transfer_fee" in t["transfers"]:
        t["transfers"]["transfer_fee"] = pd.to_numeric(t["transfers"]["transfer_fee"], errors="coerce")
    return t


def clean_games(games: pd.DataFrame, competitions=None, first_season: int = config.FIRST_SEASON) -> pd.DataFrame:
    """Keep finished games in our competitions and seasons, sorted by date."""
    competitions = competitions or config.RATING_COMPETITIONS
    g = games[games.competition_id.isin(competitions) & (games.season >= first_season)]
    g = g.dropna(subset=["date", "home_club_goals", "away_club_goals"])
    g = g[g.home_club_id != g.away_club_id]
    g = g.astype({"home_club_goals": int, "away_club_goals": int})
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def minutes_check(games: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Per (game, side): total minutes and whether it looks complete (~990)."""
    lo, hi = config.TEAM_MINUTES_RANGE
    tot = apps.groupby(["game_id", "player_club_id"]).minutes_played.sum().rename("minutes").reset_index()
    sides = pd.concat([games[["game_id", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                       games[["game_id", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    r = sides.merge(tot, left_on=["game_id", "club_id"], right_on=["game_id", "player_club_id"], how="left")
    r["minutes"] = r.minutes.fillna(0)
    r["complete"] = r.minutes.between(lo, hi)
    return r[["game_id", "club_id", "minutes", "complete"]]


def build_stints(games: pd.DataFrame, apps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stints, games_with_flag).

    stints: game_id, player_id, club_id, sign, share  (only games where BOTH sides are complete)
    games_with_flag: games plus a boolean 'has_lineups' column. Games without complete
    lineups keep their result (useful for team-only models) but are excluded from ratings.
    """
    check = minutes_check(games, apps)
    ok = check.groupby("game_id").complete.all()
    g = games.copy()
    g["has_lineups"] = g.game_id.map(ok).fillna(False).astype(bool)
    a = apps[apps.game_id.isin(g.game_id[g.has_lineups])]
    a = a.merge(g[["game_id", "home_club_id", "away_club_id"]], on="game_id")
    sign = np.where(a.player_club_id == a.home_club_id, 1, np.where(a.player_club_id == a.away_club_id, -1, 0))
    st = pd.DataFrame({"game_id": a.game_id.to_numpy(), "player_id": a.player_id.to_numpy(),
                       "club_id": a.player_club_id.to_numpy(), "sign": sign,
                       "share": a.minutes_played.to_numpy() / 90.0})
    st = st[(st.sign != 0) & (st.share > 0)]
    return st.reset_index(drop=True), g


def coverage_report(games: pd.DataFrame) -> pd.DataFrame:
    """Share of games per competition-season that have usable line-ups."""
    return (games.groupby(["competition_id", "season"])
            .agg(games=("game_id", "size"), with_lineups=("has_lineups", "mean"))
            .assign(with_lineups=lambda d: (100 * d.with_lineups).round(1)))


def prepare(tables: dict[str, pd.DataFrame], competitions=None, first_season: int = config.FIRST_SEASON) -> dict:
    """Full M1 run: clean games, quality-check minutes, build stints."""
    tables = standardise(tables)
    games = clean_games(tables["games"], competitions, first_season)
    apps = tables["appearances"][tables["appearances"].game_id.isin(games.game_id)]
    stints, games = build_stints(games, apps)
    return {"games": games, "stints": stints, "appearances": apps, "coverage": coverage_report(games)}
