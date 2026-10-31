"""
FreeAgent v2 - one-time data preparation.

What it does:
  Reads the Kaggle "Football Data from Transfermarkt" CSVs (player-scores by davidcariboo),
  keeps only what FreeAgent needs (top-5 leagues + Champions League + Europa League,
  seasons 2014 onward), checks coverage, and writes ONE small zip file to upload.

How to run (Windows / Mac / Linux):
  1. pip install pandas
  2. Download the dataset zip from Kaggle and unzip it into a folder, e.g. C:/data/transfermarkt
  3. python prepare_subset.py C:/data/transfermarkt
  4. Upload the file it creates (freeagent_subset.zip) in the Claude chat,
     and paste the summary it prints.
"""
import sys
import os
import zipfile
import pandas as pd

COMPETITIONS = ["GB1", "ES1", "IT1", "L1", "FR1", "CL", "EL"]
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]
FIRST_SEASON = 2014
OUT = "freeagent_subset.zip"
LIMIT_MB = 30


def find(folder, name):
    for ext in (".csv", ".csv.gz"):
        path = os.path.join(folder, name + ext)
        if os.path.exists(path):
            return path
    return None


def load(folder, name, required=True, **kw):
    path = find(folder, name)
    if path is None:
        if required:
            sys.exit(f"ERROR: {name}.csv not found in {folder}. Is this the unzipped Kaggle folder?")
        print(f"  (optional table {name} not found, skipping)")
        return None
    df = pd.read_csv(path, low_memory=False, **kw)
    print(f"  loaded {name}: {len(df):,} rows")
    return df


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python prepare_subset.py <folder with the Kaggle CSVs>")
    folder = sys.argv[1]
    print("Reading tables...")
    games = load(folder, "games")
    apps = load(folder, "appearances")
    transfers = load(folder, "transfers")
    players = load(folder, "players")
    clubs = load(folder, "clubs")
    comps = load(folder, "competitions")
    vals = load(folder, "player_valuations", required=False)
    lineups = load(folder, "game_lineups", required=False)

    print("\nFiltering...")
    g = games[games.competition_id.isin(COMPETITIONS) & (games.season >= FIRST_SEASON)].copy()
    g = g.dropna(subset=["home_club_goals", "away_club_goals"])
    keep_games = set(g.game_id)
    a = apps[apps.game_id.isin(keep_games)].copy()

    league_clubs = set(clubs.loc[clubs.domestic_competition_id.isin(LEAGUES), "club_id"])
    league_clubs |= set(g.loc[g.competition_id.isin(LEAGUES), "home_club_id"])
    league_clubs |= set(g.loc[g.competition_id.isin(LEAGUES), "away_club_id"])
    t = transfers.copy()
    t["transfer_date"] = pd.to_datetime(t.transfer_date, errors="coerce")
    t = t[(t.transfer_date >= f"{FIRST_SEASON}-01-01")
          & (t.to_club_id.isin(league_clubs) | t.from_club_id.isin(league_clubs))]

    keep_players = set(a.player_id) | set(t.player_id)
    p = players[players.player_id.isin(keep_players)]
    tables = {"games": g, "appearances": a, "transfers": t, "players": p,
              "clubs": clubs, "competitions": comps}
    if vals is not None:
        v = vals[vals.player_id.isin(keep_players)].copy()
        v = v[pd.to_datetime(v.date, errors="coerce") >= f"{FIRST_SEASON - 1}-01-01"]
        tables["player_valuations"] = v
    if lineups is not None:
        cols = [c for c in ["game_id", "player_id", "club_id", "type", "position", "number"] if c in lineups.columns]
        tables["game_lineups"] = lineups.loc[lineups.game_id.isin(keep_games), cols]

    print("\nCoverage check (each team should total ~990 minutes per match):")
    side = (a.groupby(["game_id", "player_club_id"]).minutes_played.sum().reset_index()
            .merge(g[["game_id", "competition_id", "season"]], on="game_id"))
    side["complete"] = side.minutes_played.between(900, 1100)
    summary = (side[side.competition_id.isin(LEAGUES)]
               .groupby(["competition_id", "season"])
               .agg(team_matches=("complete", "size"), pct_complete=("complete", "mean"),
                    avg_minutes=("minutes_played", "mean")))
    summary["pct_complete"] = (summary.pct_complete * 100).round(1)
    summary["avg_minutes"] = summary.avg_minutes.round(0)
    print(summary.to_string())

    fee = pd.to_numeric(t.transfer_fee, errors="coerce")
    epl = set(clubs.loc[clubs.domestic_competition_id == "GB1", "club_id"])
    into_epl = t[t.to_club_id.isin(epl) & (fee >= 5_000_000)]
    print("\nTransfers into current EPL clubs with fee >= EUR 5M, by year:")
    print(into_epl.groupby(into_epl.transfer_date.dt.year).size().to_string())

    print(f"\nWriting {OUT} ...")
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
            print(f"  {name}: {len(df):,} rows")
    size = os.path.getsize(OUT) / 1e6
    print(f"\nDone: {OUT} is {size:.1f} MB")
    if size > LIMIT_MB:
        print(f"WARNING: larger than {LIMIT_MB} MB. Delete player_valuations/game_lineups from the zip "
              "or tell Claude, and we'll split it.")
    print("Upload this zip in the Claude chat and paste everything printed above.")


if __name__ == "__main__":
    main()
