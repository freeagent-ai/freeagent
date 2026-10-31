"""Write reports/club_names.csv (club_id, name) so the app can show club names without the raw data.
Usage: python scripts/export_club_names.py      (a few seconds)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config, data

clubs = data.load_tables(config.DATA / "freeagent_subset.zip")["clubs"][["club_id", "name"]].drop_duplicates("club_id")
clubs.to_csv(config.REPORTS / "club_names.csv", index=False)
print(f"Saved reports/club_names.csv ({len(clubs)} clubs)")
