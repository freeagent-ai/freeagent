"""Synthetic football world in the exact Kaggle (Transfermarkt) schema.

Why it exists: every player has a KNOWN true rating, so tests can check that our
models recover the truth. Real data never tells you the right answer; this does.

World rules (deliberately simple):
- Leagues of clubs play a double round robin each season.
- Each player has a true impact theta (goal difference per 90 minutes he adds).
- Each match, a club starts its 11 best players (with noise) and makes 3 substitutions.
- Goals are Poisson: log(rate) = base + home + kappa * (minutes-weighted strength gap).
- Every summer ~20% of players move club (also across leagues), so player and
  club effects can be separated, exactly as in the real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]


def make_world(n_leagues: int = 2, clubs_per_league: int = 10, squad: int = 18,
               seasons: tuple = (2014, 2015, 2016, 2017, 2018, 2019), move_rate: float = 0.2,
               sigma: float = 0.2, kappa: float = 0.35, home: float = 0.15, base: float = 0.25,
               seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    leagues = ["GB1", "ES1", "IT1", "L1", "FR1"][:n_leagues]
    clubs = pd.DataFrame([{"club_id": 100 + li * 50 + c, "name": f"Club {lg}-{c}", "domestic_competition_id": lg}
                          for li, lg in enumerate(leagues) for c in range(clubs_per_league)])
    n_players = len(clubs) * squad
    theta = rng.normal(0, sigma, n_players)
    birth_year = rng.integers(1988, 2000, n_players)
    players = pd.DataFrame({
        "player_id": np.arange(1, n_players + 1),
        "name": [f"Player {i}" for i in range(1, n_players + 1)],
        "date_of_birth": [f"{y}-06-15" for y in birth_year],
        "position": rng.choice(POSITIONS, n_players, p=[0.1, 0.35, 0.35, 0.2]),
        "height_in_cm": rng.normal(181, 6, n_players).round(),
    })
    club_of = np.repeat(clubs.club_id.to_numpy(), squad)       # starting allocation
    games, apps, transfers = [], [], []
    game_id, app_id = 1, 1
    for s_idx, season in enumerate(seasons):
        if s_idx > 0:                                          # summer transfer window
            # Players move in swapped pairs so every squad keeps its size.
            movers = rng.choice(n_players, int(move_rate * n_players) // 2 * 2, replace=False)
            new_club = {}
            for p, q in zip(movers[::2], movers[1::2]):
                if club_of[p] != club_of[q]:
                    new_club[p], new_club[q] = club_of[q], club_of[p]
            for p, new in new_club.items():
                age = season - birth_year[p]
                fee = np.exp(15.5 + 4.0 * theta[p] - 0.08 * (age - 25) ** 2 / 4 + rng.normal(0, 0.5))
                transfers.append({"player_id": p + 1, "transfer_date": f"{season}-07-15",
                                  "transfer_season": f"{season % 100:02d}/{(season + 1) % 100:02d}",
                                  "from_club_id": club_of[p], "to_club_id": new,
                                  "transfer_fee": round(fee, -4),
                                  "market_value_in_eur": round(fee * np.exp(rng.normal(0, 0.3)), -4)})
                club_of[p] = new
        for lg in leagues:
            ids = clubs.club_id[clubs.domestic_competition_id == lg].to_numpy()
            fixtures = [(h, a) for h in ids for a in ids if h != a]
            rng.shuffle(fixtures)
            for k, (h, a) in enumerate(fixtures):
                date = pd.Timestamp(f"{season}-08-10") + pd.Timedelta(days=int(k * 270 / len(fixtures)))
                strength, rows = {}, []
                for club in (h, a):
                    sq = np.flatnonzero(club_of == club)
                    order = sq[np.argsort(-(theta[sq] + rng.normal(0, 0.15, len(sq))))]
                    starters, bench = order[:11], order[11:]
                    minutes = {p: 90.0 for p in starters}
                    for out, inn in zip(rng.choice(starters, 3, replace=False), bench[:3]):
                        m = int(rng.integers(55, 86))
                        minutes[out], minutes[inn] = float(m), float(90 - m)
                    strength[club] = sum(theta[p] * m / 90 for p, m in minutes.items())
                    rows += [(p, club, m) for p, m in minutes.items()]
                gap = strength[h] - strength[a]
                hg = rng.poisson(np.exp(base + home + kappa * gap))
                ag = rng.poisson(np.exp(base - kappa * gap))
                scorer_goals = {}
                for club, n in ((h, hg), (a, ag)):
                    on = [(p, m) for p, c, m in rows if c == club]
                    w = np.array([m * np.exp(3 * theta[p]) for p, m in on]); w /= w.sum()
                    for _ in range(n):
                        p = on[rng.choice(len(on), p=w)][0]
                        scorer_goals[p] = scorer_goals.get(p, 0) + 1
                for p, club, m in rows:
                    apps.append({"appearance_id": app_id, "game_id": game_id, "player_id": p + 1,
                                 "player_club_id": club, "date": date.strftime("%Y-%m-%d"),
                                 "competition_id": lg, "minutes_played": int(round(m)),
                                 "goals": scorer_goals.get(p, 0), "assists": 0,
                                 "yellow_cards": 0, "red_cards": 0})
                    app_id += 1
                games.append({"game_id": game_id, "competition_id": lg, "season": season,
                              "date": date.strftime("%Y-%m-%d"), "home_club_id": h, "away_club_id": a,
                              "home_club_goals": hg, "away_club_goals": ag})
                game_id += 1
    competitions = pd.DataFrame({"competition_id": leagues, "name": leagues})
    return {"games": pd.DataFrame(games), "appearances": pd.DataFrame(apps),
            "players": players, "clubs": clubs, "competitions": competitions,
            "transfers": pd.DataFrame(transfers),
            "truth": pd.DataFrame({"player_id": players.player_id, "theta": theta})}


def make_valuations(world: dict, seed: int = 0) -> pd.DataFrame:
    """Fake market values for the synthetic world (it has none): every 6 months, value rises with true
    quality and with age up to 27, then falls; plus noise. Only used to test Option B code paths."""
    rng = np.random.default_rng(seed)
    pl = world["players"].merge(world["truth"], on="player_id")
    pl["date_of_birth"] = pd.to_datetime(pl.date_of_birth)
    dates = pd.date_range("2013-01-01", "2022-12-31", freq="6MS")
    rows = []
    for d in dates:
        age = (d - pl.date_of_birth).dt.days / 365.25
        logv = 6.5 + 2.0 * pl.theta - 0.01 * (age - 27) ** 2 + rng.normal(0, 0.15, len(pl))
        rows.append(pd.DataFrame({"player_id": pl.player_id, "date": d, "market_value_in_eur": 10 ** logv}))
    return pd.concat(rows, ignore_index=True)
