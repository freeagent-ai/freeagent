"""Project-wide settings. Change scope here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Leagues used to RATE players (top 5 + second tiers + European cups).
TOP_LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]
SECOND_TIERS = ["GB2", "ES2", "IT2", "L2", "FR2"]
CUPS = ["CL", "EL"]
RATING_COMPETITIONS = TOP_LEAGUES + SECOND_TIERS + CUPS

# Transfers we EVALUATE: into Premier League clubs, fee >= 5M euro.
TARGET_LEAGUE = "GB1"
MIN_FEE_EUR = 5_000_000
FIRST_SEASON = 2014          # Transfermarkt season label = starting year (2014 = 2014/15)
EVAL_WINDOWS = list(range(2017, 2025))   # summer windows evaluated walk-forward

# Data-quality rule: one team's minutes in a match should total ~990 (11 x 90).
TEAM_MINUTES_RANGE = (900, 1100)

SEED = 42
