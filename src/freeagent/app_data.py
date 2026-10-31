"""Data and text helpers for the Streamlit app (no Streamlit imports here, so they are testable)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config, value as V

SIGNINGS_FILE = "m5_signing_points.csv"
SEASON_FILE = "season_predictions.csv"
CLUB_NAMES_FILE = "club_names.csv"
LEAGUE_NAMES = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}
VERDICT_COLOURS = {"Overpaid": "#B03A2E", "Fair": "#7F8C8D", "Best Buy": "#1E8449", "Too close to call": "#5D6D7E",
                   "Not rated": "#34495E"}
CONFIDENT_MIN = 0.5      # below this the top verdict is hardly more likely than the others: do not call it
VERDICT_CHOICES = ["Overpaid", "Fair", "Best Buy", "Too close to call", "Not rated"]


def shown_verdict(verdict: str, confidence: float, known: bool = True) -> str:
    """No rating -> we know nothing about him, so we do not judge (a 'Fair' would wrongly sound like a verdict)."""
    if not known:
        return "Not rated"
    return verdict if confidence >= CONFIDENT_MIN else "Too close to call"


def season_label(y: int) -> str:
    return f"{int(y)}/{str(int(y) + 1)[-2:]}"


def load_signings(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.REPORTS / SIGNINGS_FILE)
    df["fee_m"] = df.transfer_fee / 1e6
    df["season_label"] = df.season.map(season_label)
    df["claimed_points"] = df.claimed_points.astype(float)
    df["shown_verdict"] = [shown_verdict(v, c, bool(k)) for v, c, k in zip(df.verdict, df.confidence, df.known_player)]
    df["label"] = (df.player_name.fillna("Unknown") + " - " + df.to_club_name.fillna("?") +
                   " (" + df.season_label + ")")
    df = add_player_extras(df)
    return df.sort_values(["season", "fee_m"], ascending=[False, False]).reset_index(drop=True)


def add_player_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Country flag and a freely licensed photo with its credit, when the lookup files exist
    (scripts/export_app_extras.py and scripts/fetch_player_photos.py). Missing files -> empty columns."""
    from . import flags
    info = load_report("player_info.csv")
    ctry = df.player_id.map(info.drop_duplicates("player_id").set_index("player_id").country) if info is not None else None
    df["country"] = ctry if ctry is not None else np.nan
    df["flag"] = df.country.map(flags.code)
    ph = load_report("player_photos.csv")
    if ph is not None and len(ph):
        ph = ph.drop_duplicates("player_id").set_index("player_id").fillna("")
        df["photo"] = df.player_id.map(ph.thumb_url).fillna("")
        df["photo_page"] = df.player_id.map(ph.page_url).fillna("")
        credit = ("Photo: " + ph.artist.str.slice(0, 60) + " · " + ph.license + " · Wikimedia Commons")
        df["photo_credit"] = df.player_id.map(credit.where(ph.thumb_url != "", "")).fillna("")
    else:
        df["photo"], df["photo_page"], df["photo_credit"] = "", "", ""
    return df


def load_report(name: str, **kw) -> pd.DataFrame | None:
    p = config.REPORTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def filter_signings(df, seasons=None, leagues=None, verdicts=None, min_fee=0.0, text="") -> pd.DataFrame:
    m = df.fee_m >= min_fee
    if seasons:
        m &= df.season.isin(seasons)
    if leagues:
        m &= df.to_league.isin(leagues)
    if verdicts:
        m &= df.shown_verdict.isin(verdicts)
    if text:
        m &= df.label.str.contains(text, case=False, regex=False)
    return df[m]


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Player": df.player_name, "Club": df.to_club_name, "Season": df.season_label,
        "Fee (EUR M)": df.fee_m.round(1), "Chance of becoming a regular": (100 * df.p_regular).round(0).astype(int),
        "Points added (best guess)": df.points_mean.round(2),
        "Range (90%)": [f"{a:+.1f} to {b:+.1f}" for a, b in zip(df.points_p05, df.points_p95)],
        "Price claims (pts)": df.claimed_points.round(2),
        "Verdict": df.shown_verdict, "Confidence (%)": (100 * df.confidence).round(0).astype(int)})


def verdict_sentence(r) -> str:
    claim = (f"The price tag of EUR {r.fee_m:.1f}M claims about {r.claimed_points:.1f} league points a season "
             f"(market rate: {r.price_b_per_100m:.1f} points per EUR 100M of squad value). ")
    if r.shown_verdict == "Not rated":
        return claim + "We have no match history for this player, so we do not judge the fee."
    est = (f"We estimate he adds {r.points_mean:+.1f} points "
           f"(90% range {r.points_p05:+.1f} to {r.points_p95:+.1f}). ")
    if r.shown_verdict == "Too close to call":
        return claim + est + (f"The three outcomes are too close to call (the most likely, '{r.verdict}', is only "
                              f"{100 * r.confidence:.0f}% likely).")
    return claim + est + f"So the model leans '{r.verdict}' with {100 * r.confidence:.0f}% probability."


def what_happened(r) -> str:
    played = 100 * r.actual_share
    outcome = "became a regular" if r.regular == 1 else "did not become a regular"
    return (f"In reality he played {played:.0f}% of {r.to_club_name}'s league minutes in {r.season_label} "
            f"and {outcome}. Before the season we gave him a {100 * r.p_regular:.0f}% chance of becoming a regular.")


def calculator(fee_m: float, gain_mean: float, gain_sd: float, points_per_100m: float, tolerance: float,
               seed: int = config.SEED) -> dict:
    """Same verdict code as the research pipeline, applied to a normal guess of the points gain."""
    draws = np.random.default_rng(seed).normal(gain_mean, max(gain_sd, 1e-6), 20000)
    out = V.verdict_probs(draws, fee_m * 1e6, points_per_100m / 100.0, tolerance)
    out["draws"] = draws
    return out


def load_season(path=None) -> pd.DataFrame | None:
    """Season predictions (top half / bottom half, made before each season) with club and league names."""
    p = path or config.REPORTS / SEASON_FILE
    if not p.exists():
        return None
    df = pd.read_csv(p)
    names = load_report(CLUB_NAMES_FILE)
    if "club_name" not in df:
        df["club_name"] = (df.club_id.map(names.set_index("club_id").name) if names is not None
                           else pd.Series(np.nan, index=df.index))
    df["club_name"] = df.club_name.fillna("Club " + df.club_id.astype(str))
    df["league_name"] = df.league.map(LEAGUE_NAMES).fillna(df.league)
    df["season_label"] = df.season.map(season_label)
    df["correct"] = (df.pred_top_half == df.actual_top_half).astype(int)
    return df


def season_table(df: pd.DataFrame, season: int, league: str) -> pd.DataFrame:
    """One league-season: our pre-season forecast next to the real final table."""
    g = df[(df.season == season) & (df.league == league)].sort_values("pred_points", ascending=False)
    half = lambda x: np.where(x == 1, "Top half", "Bottom half")
    return pd.DataFrame({
        "Club": g.club_name, "Predicted points": g.pred_points.round(1),
        "Chance of top half (%)": (100 * g.p_top_half).round(0).astype(int),
        "Chance of relegation (%)": (100 * g.p_relegation).round(0).astype(int),
        "We predicted": half(g.pred_top_half), "Actual": half(g.actual_top_half),
        "Right?": np.where(g.correct == 1, "Yes", "No"),
        "Actual points": g.actual_points.astype(int), "Actual position": g.actual_rank.astype(int)})


def headline_numbers() -> dict:
    """The key results shown on the home page, read from the saved reports (never typed in by hand)."""
    out = {}
    m8 = load_report("m8_summary.csv", index_col=0)
    if m8 is not None:
        r = m8.loc["Ensemble: logistic + boosting (ours)"]
        out.update(m8_accuracy=float(r.accuracy), m8_conf_accuracy=float(r.confident_accuracy),
                   m8_coverage=float(r.coverage), m8_base=float(m8.loc["Majority class (base rate)"].accuracy),
                   m8_n=int(r.n))
    rq = load_report("rq1_summary.csv", index_col=0)
    if rq is not None:
        out.update(rq1_hybrid=float(rq.loc["Hybrid: player + Elo (ours)"].log_loss),
                   rq1_elo=float(rq.loc["Elo"].log_loss), rq1_n=int(rq.n.iloc[0]))
    ss = load_report("season_summary.csv", index_col=0)
    if ss is not None:
        ours = [i for i in ss.index if i.startswith("OURS")][0]
        out.update(season_accuracy=float(ss.loc[ours].accuracy), season_lo=float(ss.loc[ours].ci_lo),
                   season_hi=float(ss.loc[ours].ci_hi), season_n=int(ss.loc[ours].n),
                   season_last=float(ss.loc["Same half as last season"].accuracy) if "Same half as last season" in ss.index else np.nan,
                   season_value=float(ss.loc["Squad market value"].accuracy) if "Squad market value" in ss.index else np.nan)
    pt = load_report("season_points_test.csv")
    if pt is not None:
        r = pt[pt.measure.str.startswith("mean absolute")].iloc[0]
        out.update(points_mae=float(r.ours), points_mae_last=float(r.last_season),
                   points_verdict=str(r.league_season_verdict))
    s = load_signings() if (config.REPORTS / SIGNINGS_FILE).exists() else None
    if s is not None:
        out.update(n_signings=len(s), verdict_counts=s.shown_verdict.value_counts().to_dict())
    return out


# ================================================================== 2026/27: the live forecast (scripts/forecast_2026.py)
FORECAST_TABLE = "forecast_2026_table.csv"
FORECAST_SIGNINGS = "forecast_2026_signings.csv"


def load_forecast() -> pd.DataFrame | None:
    df = load_report(FORECAST_TABLE)
    if df is None:
        return None
    df["league_name"] = df.league.map(LEAGUE_NAMES).fillna(df.league)
    return df.sort_values(["league", "pred_rank"]).reset_index(drop=True)


def forecast_meta() -> dict:
    m = load_report("forecast_2026_meta.csv", index_col=0)
    return {} if m is None else m["value"].to_dict()


def forecast_table(df: pd.DataFrame, league: str) -> pd.DataFrame:
    g = df[df.league == league].sort_values("pred_rank")
    return pd.DataFrame({
        "#": g.pred_rank.astype(int), "Club": g.club_name + np.where(g.promoted == 1, " ↑", ""),
        "Points": g.mean_points.round(1), "Range": [f"{a:.0f}-{b:.0f}" for a, b in zip(g.points_p10, g.points_p90)],
        "Title": (100 * g.p_title).round(0).astype(int), "Top 4": (100 * g.p_top4).round(0).astype(int),
        "Top half": (100 * g.p_top_half).round(0).astype(int),
        "Relegation": (100 * g.p_relegation).round(0).astype(int), "club_id": g.club_id})


def load_forecast_signings() -> pd.DataFrame | None:
    df = load_report(FORECAST_SIGNINGS)
    if df is None or len(df) == 0:
        return None
    df["fee_m"] = df.transfer_fee / 1e6
    df["season_label"] = df.season.map(season_label)
    for c in ["points_mean", "points_p05", "points_p95", "claimed_points", "confidence", "p_overpaid", "p_fair",
              "p_best_buy", "price_b_per_100m"]:
        if c not in df:
            df[c] = np.nan
    rated = df.verdict.ne("Not rated")
    df["shown_verdict"] = [shown_verdict(v, c, bool(k)) if r else "Not rated"
                           for v, c, k, r in zip(df.verdict, df.confidence.fillna(0), df.known_player, rated)]
    df["label"] = (df.player_name.fillna("Unknown") + " - " + df.to_club_name.fillna("?") +
                   " (from " + df.from_club_name.fillna("?") + ")")
    df = add_player_extras(df)
    return df.sort_values("fee_m", ascending=False).reset_index(drop=True)


def forecast_sentence(r) -> str:
    """Scouting line for a signing whose first season has not happened yet."""
    reg = f"We give him a {100 * r.p_regular:.0f}% chance of playing at least half of the league minutes this season. "
    if r.shown_verdict == "Not rated":
        why = str(getattr(r, "note", "") or "")
        why = why if why and why != "nan" else "we have no match history for him"
        return reg + f"We do not judge the fee: {why}."
    return reg + verdict_sentence(r)


# ================================================================== Transfer Lab: any player, any club, any fee
def load_lab() -> dict | None:
    sq, cv, pl = load_report("lab_squads.csv"), load_report("lab_curves.csv"), load_report("lab_players.csv")
    if sq is None or cv is None or pl is None:
        return None
    pl = add_player_extras(pl.rename(columns={}).copy())
    pl["label"] = pl.player_name.fillna("?") + " · " + pl.position.fillna("?") + " · " + pl.club_name.fillna("")
    return {"squads": sq, "curves": cv, "players": pl.reset_index(drop=True)}


def lab_estimate(lab: dict, club_id: int, player_id: int, share: float, fee_m: float, points_per_100m: float,
                 tolerance: float = V.TOLERANCE_POINTS, n: int = 4000, seed: int = config.SEED) -> dict:
    """What would this player add to this club? Same maths as M5 v2 (weakest players lose their minutes first) on
    the saved 2026/27 squad and points curve. share = his expected share of the league minutes (0-1)."""
    from . import simulate as M
    p = lab["players"].set_index("player_id").loc[player_id]
    sq = lab["squads"]
    club = sq[sq.club_id == club_id]
    rest = M.without_player(club, club_id, player_id)
    order = M.displacement_order(rest, p.position)
    rng = np.random.default_rng(seed)
    r_new = rng.normal(p.rating, max(float(p.rating_sd), 1e-9), n)
    delta = share * r_new - M.displaced_mass(order, np.array([share]))[0]
    cv = lab["curves"][lab["curves"].club_id == club_id].sort_values("delta")
    f = lambda x: np.interp(x, cv.delta, cv.points)
    gain = f(delta) - f(0.0)
    out = V.verdict_probs(gain, fee_m * 1e6, points_per_100m / 100.0, tolerance)
    # who loses minutes, weakest first
    same = rest[rest.position == p.position].sort_values("rating", kind="stable")
    other = rest[rest.position != p.position].sort_values("rating", kind="stable")
    left, lost = share, []
    for r in pd.concat([same, other]).itertuples():
        if left <= 1e-9:
            break
        take = min(left, r.share)
        lost.append((str(getattr(r, "player_name", "") or f"Player {r.player_id}"), take, float(r.rating)))
        left -= take
    out.update(points_mean=float(gain.mean()), points_p05=float(np.percentile(gain, 5)),
               points_p95=float(np.percentile(gain, 95)), draws=gain, displaced=lost,
               already_here=bool((club.player_id == player_id).any()), rating=float(p.rating))
    return out
