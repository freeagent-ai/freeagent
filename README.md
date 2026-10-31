# FreeAgent — Buying Points, Not Names

Pricing football transfers in expected league points, using cross-league Bayesian
player ratings, a Poisson/Dixon-Coles match model and Monte Carlo season simulation.

**What it is for:** the live 2026/27 season. `scripts/forecast_2026.py` forecasts every top-five league table,
judges the summer 2026 signings before they have played, and builds the Transfer Lab ("what would player X add
to club Y for fee Z?"). **Past seasons (2019-2025) are the track record:** the same method run blind, each season
predicted only from earlier data, then checked against what really happened. That is where every accuracy number
below comes from; the 2026/27 forecast itself can only be scored in May 2027.

## Status

| Module | File | State |
| --- | --- | --- |
| M1 Data pipeline | `src/freeagent/data.py` | Done, tested |
| M2 ML prior | `src/freeagent/prior.py` | Done, tested, real-data results |
| M3 Bayesian plus-minus | `src/freeagent/impact.py` | Done, tested |
| M4 Match model + baselines | `src/freeagent/match.py` | Done, tested, real-data results (RQ1) |
| M5 Season simulator | `src/freeagent/simulate.py` | Done, tested, run on real data |
| M6 Valuation | `src/freeagent/value.py` | Done, tested, run on real data |
| M7 Validation | `scripts/m7_validate_verdicts.py` | Done (result: no detectable link, low power) |
| M8 Signing-success classifier | `src/freeagent/success.py` | Done, tested, real-data results (accuracy headline) |

## Real-data results so far (Transfermarkt, top-5 leagues)

| Result | Number | Script |
| --- | --- | --- |
| RQ1: hybrid (player + Elo) vs Elo, log loss | 0.9930 vs 0.9960, significantly better | `scripts/rq1_match_backtest.py` |
| M2: player model with prior vs without | 1.0013 vs 1.0035, significantly better | `scripts/m2_prior_backtest.py` |
| M8: will a EUR 5M+ signing become a regular? all signings | 64.2% accuracy (base rate 51.4%) | `scripts/m8_signing_success.py` |
| M8: confident predictions (p >= 0.6 or <= 0.4) | 71.1% accuracy on 60.4% of signings | `scripts/m8_signing_success.py` |
| M6: price of a league point | 1.3 points per EUR 100M within clubs vs 6.7 between clubs (not well identified) | `scripts/m5_points_per_signing.py` |
| Season: top or bottom half, predicted before the season (680 club-seasons) | 77.1% (95% CI 73.8-80.1%), tied with strong baselines; points error 8.6 vs 9.2 | `scripts/season_predict.py` |
| Season, with the promoted-club rule (used for 2026/27) | 77.4%, points error 8.6, relegation Brier 0.116 vs 0.119 | `scripts/season_predict.py --promoted relegated` |
| M5 v2 bias fix (30 Oct 2026): 'Overpaid' among signings of the strongest quarter of clubs | 46% (v1) -> 6% (v2) | `scripts/m5_points_per_signing.py` |
| M6: verdicts at the low price (default, v2) | 3% Overpaid, 81% Fair, 16% Best Buy; our estimates decide about 32% of verdicts | `scripts/m6_shuffle_test.py` |
| M6: verdicts at the high price (v2) | 43% Overpaid; a fee-only rule reproduces 85% of them | `scripts/m6_shuffle_test.py` |
| M7: predicted points vs realised on/off (top minus bottom third, v2) | -0.042 (95% CI -0.136 to +0.052), not significant (v1: -0.027) | `scripts/m7_validate_verdicts.py` |

`src/freeagent/synthetic.py` builds a fake football world in the same format as the
real data, where every player's true rating is known. Tests use it to check that the
models recover the truth.

## Setup (one time)

```
pip install -r requirements.txt
```
All results were produced with Python 3.15.1 and the exact package versions in `requirements.txt` (full list in `requirements-lock.txt`).

## Run the tests

```
pytest                      # if pytest is installed
python tests/run_tests.py   # works without pytest
```

## RQ1 backtest (go/no-go)

```
python scripts/rq1_match_backtest.py --synthetic   # quick check on fake data
python scripts/rq1_match_backtest.py               # real data in data/freeagent_subset.zip
```

## M2 and M8 (real data)

```
python scripts/m2_prior_backtest.py      # about 4 minutes
python scripts/m8_signing_success.py     # about 1 minute
python scripts/m8_improve.py             # round 2: new features + comparison with published model families, about 2 minutes
python scripts/m8_compare.py             # head-to-head against 11 standard model families, with significance tests
python scripts/robustness_checks.py      # label threshold, by position, rating half-life (M8); add --parts halflife_rq1 for the slow match-model check
python scripts/season_predict.py         # Option A: top half or bottom half, predicted before the season (tests the whole chain)
python scripts/season_points_test.py     # are our points forecasts significantly better than last season's points? (seconds)
python scripts/value_rise.py             # Option B: will a player's market value rise in the next 12 months?
```

## M5, M6, M7 (real data)

```
python scripts/m5_points_per_signing.py   # about 7 minutes, writes reports/m5_signing_points.csv (method v2)
python scripts/m5_points_per_signing.py --method v1   # the original method, for the before/after comparison
python scripts/m6_shuffle_test.py         # control: do verdicts depend on more than the fee?
python scripts/m7_validate_verdicts.py    # about 1 minute, needs the file above
```

## The live forecast (2026/27)

```
python scripts/forecast_2026.py           # about 14 minutes: forecast_2026_*.csv and lab_*.csv in reports/
```
Same frozen settings as the track record, trained on all data up to the July 2026 snapshot. 2026/27 line-ups
(relegation, promotion, play-offs) are listed at the top of the script.

## The app

```
streamlit run app.py
```
One scrolling page with a sticky section menu (Forecast 26/27, Signings 2026, Transfer Lab, Track record, Method); dark glass theme with Unbounded, JetBrains Mono and Manrope from Google Fonts.
Run once: `python scripts/export_app_extras.py` (club names and player countries for the flags, offline) and
`python scripts/fetch_player_photos.py` (needs internet; finds freely licensed player photos on Wikimedia Commons and
stores the author and licence, which the app shows next to each photo). No league logos are used (they are trademarks). The app only reads the
files in `reports/` (made by the scripts above), so it starts instantly and needs no raw data.
Any verdict whose top outcome is under 50% likely is shown as "Too close to call".

## Data

Put `freeagent_subset.zip` (made by `prepare_subset.py` from the Kaggle
"Football Data from Transfermarkt" dataset) in `data/`. Raw data is never committed.

## Layout

```
src/freeagent/   one file per module (see table above)
tests/           tests for every module
data/            input data (not in git)
reports/         every number and figure used in the paper
```
