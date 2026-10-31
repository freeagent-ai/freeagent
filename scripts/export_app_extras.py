"""Write small lookup files the app needs, so it never has to read the raw data:
  reports/club_names.csv   club_id, name
  reports/player_info.csv  player_id, country, position  (signings 2019-2025, summer 2026 signings, Transfer Lab players)
Usage: python scripts/export_app_extras.py      (a few seconds, offline)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data, flags

t = data.load_tables(config.DATA / "freeagent_subset.zip")
clubs = t["clubs"][["club_id", "name"]].drop_duplicates("club_id")
clubs.to_csv(config.REPORTS / "club_names.csv", index=False)
print(f"Saved reports/club_names.csv ({len(clubs)} clubs)")

files = ["m5_signing_points.csv", "forecast_2026_signings.csv", "lab_players.csv"]
sig = pd.concat([pd.read_csv(config.REPORTS / f, usecols=["player_id"]) for f in files if (config.REPORTS / f).exists()])
p = t["players"]
p = p[p.player_id.isin(sig.player_id)].drop_duplicates("player_id")
country = p["country_of_citizenship"] if "country_of_citizenship" in p else pd.Series(index=p.index, dtype=object)
out = pd.DataFrame({"player_id": p.player_id, "country": country,
                    "position": p.get("position"), "sub_position": p.get("sub_position")})
out.to_csv(config.REPORTS / "player_info.csv", index=False)
missing = sorted(set(out.country.dropna()) - set(flags.CODES))
print(f"Saved reports/player_info.csv ({len(out)} players, {out.country.notna().mean():.0%} with a country)")
if missing:
    print("Countries without a flag code (shown without a flag):", ", ".join(missing))
