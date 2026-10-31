"""FreeAgent: Buying Points, Not Names. One scrolling page with a sticky section menu.   Run:  streamlit run app.py

Order of the page = what the product is for:
  01 Forecast 2026/27   the season being played now, forecast before it started (nobody knows the answer yet)
  02 Signings 2026      this summer's big signings, judged before they have played
  03 Transfer Lab       any player, any club, any fee: what would he add?
  04 Track record       the same method run blind on 2019-2025 and checked against what really happened
  05 Method
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from freeagent import app_data as D, theme as T

st.set_page_config(page_title="FreeAgent", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
st.markdown(T.CSS, unsafe_allow_html=True)


def html(s, where=st):
    where.markdown(s, unsafe_allow_html=True)


@st.cache_data
def signings():
    return D.load_signings()


@st.cache_data
def season():
    return D.load_season()


@st.cache_data
def forecast():
    return D.load_forecast()


@st.cache_data
def signings_now():
    return D.load_forecast_signings()


@st.cache_data
def moves():
    return D.load_report("forecast_2026_moves.csv")


@st.cache_resource
def lab():
    return D.load_lab()


@st.cache_data
def report(name, index_col=None):
    return D.load_report(name, index_col=index_col)


def interval_chart(r):
    """Our range for the points a signing adds, against what the fee claims."""
    fig, ax = plt.subplots(figsize=(8, 2.4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    c = T.VERDICT.get(r.shown_verdict, T.MUTED)
    ax.axvspan(r.claimed_points - 1.0, r.claimed_points + 1.0, color=T.MUTED, alpha=0.12, label="'Fair' zone around the price")
    ax.axvline(r.claimed_points, color=T.MUTED, ls="--", lw=1.5, label=f"Price claims {r.claimed_points:.1f} pts")
    ax.axvline(0, color=T.MUTED, lw=0.6, alpha=0.6)
    ax.hlines(0, r.points_p05, r.points_p95, color=c, lw=9, alpha=0.85, label="Our 90% range")
    ax.plot([r.points_mean], [0], "o", color="#FFFFFF", mec=c, mew=2.5, ms=11, label=f"Best guess {r.points_mean:+.1f}")
    ax.set_yticks([])
    ax.set_xlabel("Extra league points per season", color=T.MUTED)
    ax.tick_params(colors=T.MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T.MUTED)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(T.MUTED)
    fig.tight_layout()
    return fig


def scouting_report(r, past: bool):
    """Verdict, probability bar, range chart and the plain-English reading for one signing."""
    lcol, rcol = st.columns([2, 3])
    html(T.player_card(r), lcol)
    if r.shown_verdict == "Not rated":
        html(T.glass(f"<div class='fa-label'>Verdict</div><div style='margin:12px 0'>{T.badge(r.shown_verdict)}</div>"
                     f"<div class='fa-sub'>Chance he becomes a regular: {100 * r.p_regular:.0f}%</div>"), lcol)
    else:
        html(T.glass(f"<div class='fa-label'>Verdict</div><div style='margin:12px 0'>{T.badge(r.shown_verdict)}</div>"
                     f"<div class='fa-sub'>{100 * r.confidence:.0f}% for the most likely outcome · price claims "
                     f"{r.claimed_points:.1f} pts</div>"), lcol)
        html(T.glass("<div class='fa-label'>Overpaid · Fair · Best Buy</div>" +
                     T.prob_bar(r.p_overpaid, r.p_fair, r.p_best_buy)), rcol)
        rcol.pyplot(interval_chart(r))
    text = D.verdict_sentence(r) if past else D.forecast_sentence(r)
    after = (D.what_happened(r) if past else
             "His first season is being played now, so nobody knows the answer yet: check back in May 2027.")
    html(T.glass(f"<p class='fa-sub' style='font-size:1rem;color:#DCE2F5'>{T.esc(text)}</p>"
                 f"<p class='fa-sub' style='font-size:1rem;color:#DCE2F5;margin-bottom:0'>{T.esc(after)}</p>"))


h = D.headline_numbers()
pct = lambda x: f"{100 * x:.1f}%"
fc, meta, now = forecast(), D.forecast_meta(), signings_now()

# ================================================================== menu + hero
html(T.nav())
left, right = st.columns([3, 2], gap="large")
with left:
    html(T.hero("Live forecast", "2026/27 · top-5 leagues", "Tested on 2019-2025",
                "Before a ball was kicked in 2026/27, FreeAgent rated every player, rebuilt every squad with the summer's "
                "transfers and played the season 5,000 times. Here is that forecast, this summer's signings judged before "
                "they have played, and the proof that the method works."))
    html(T.league_badges(D.LEAGUE_NAMES))
with right:
    items = []
    if fc is not None:
        items.append(("Clubs forecast for 2026/27", f"{len(fc)}"))
    if now is not None:
        items.append(("Summer 2026 signings judged", f"{len(now)}"))
    if "season_accuracy" in h:
        items.append(("Track record: season forecast", pct(h["season_accuracy"])))
    if "m8_accuracy" in h:
        items.append(("Track record: becomes regular", f"{pct(h['m8_accuracy'])} · {pct(h['m8_conf_accuracy'])} sure"))
    if "points_mae" in h:
        items.append(("Points error vs last season", f"{h['points_mae']:.1f} vs {h['points_mae_last']:.1f}"))
    html("<div style='height:56px'></div>")
    html(T.rows(items, highlight=0))
if now is not None and len(now):
    html("<div class='fa-label' style='margin:34px 0 14px'>Spotlight · biggest signings of summer 2026</div>")
    for _i, (_col, _r) in enumerate(zip(st.columns(3), now.head(3).itertuples(index=False))):
        html(T.spotlight_card(_r, _i), _col)
html(T.ticker(["2026/27 forecast", "Bayesian plus-minus", "Dixon-Coles", "Monte Carlo", "Gradient boosting",
               "Walk-forward testing", "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]))

html(T.section("00", "ways", "Three ways in", "Start here", "The short version. Each card jumps to its section."))
w1, w2, w3 = st.columns(3)
html(T.way("01", "Forecast", "How will 2026/27 end?",
           "Every club's points, title, top-4 and relegation chances, frozen before the season started.",
           "See the forecast", "forecast", T.ACCENT), w1)
html(T.way("02", "Signings", "Were this summer's fees worth it?",
           "The summer 2026 signings judged before they have played: regular or not, points added, verdict.",
           "Judge the window", "signings", T.CYAN), w2)
html(T.way("03", "Transfer Lab", "Should my club sign him?",
           "Pick a club, a player and a fee. See the points he would add and who would lose his place.",
           "Open the lab", "lab", T.VIOLET), w3)

# ================================================================== 01 forecast 2026/27
html(T.section("01", "forecast", "Forecast 2026/27", "Before a ball was kicked",
               f"Data up to {meta.get('last_match', 'May 2026')}, summer transfers up to {meta.get('last_transfer', 'July 2026')}. "
               "The season is being played now: this forecast was frozen before it began, so you can hold it against "
               "the live table. Nothing was re-tuned for 2026: these are the exact settings tested on 2019-2025."))
if fc is None:
    st.info("No 2026/27 forecast yet. Run `python scripts/forecast_2026.py` to create it.")
else:
    lg = st.selectbox("League", list(D.LEAGUE_NAMES), format_func=lambda x: D.LEAGUE_NAMES.get(x, x), key="fc_league")
    g = fc[fc.league == lg]
    top = g.sort_values("p_title", ascending=False).head(3)
    down = g.sort_values("p_relegation", ascending=False).head(1).iloc[0]
    cols = st.columns(4)
    for i, (col, r) in enumerate(zip(cols, top.itertuples(index=False))):
        html(T.stat_card("Title favourite" if i == 0 else f"Title chance #{i + 1}", f"{100 * r.p_title:.0f}%",
                         f"{r.club_name} · {r.mean_points:.0f} pts", accent=(i == 0)), col)
    html(T.stat_card("Most likely to go down", f"{100 * down.p_relegation:.0f}%", f"{down.club_name} · {down.mean_points:.0f} pts"),
         cols[3])
    html(f"<div class='fa-label' style='margin:18px 0 10px'>{T.esc(D.LEAGUE_NAMES.get(lg, lg))} 2026/27 · predicted table "
         "(green bar = top 4, red = relegation zone, ↑ = promoted)</div>")
    tbl = D.forecast_table(fc, lg)
    html(T.forecast_table(tbl.drop(columns="club_id")))
    st.caption("Points = average over 5,000 simulated seasons; the 80% range is where his club ends in 8 of 10 simulations. "
               "Promoted clubs (↑) get the average strength of the clubs they replaced, because our data has no "
               "second-division matches.")
    mv = moves()
    if mv is not None:
        club = st.selectbox("What changed this summer?", list(tbl.Club), key="fc_club")
        cid = int(tbl[tbl.Club == club].club_id.iloc[0])
        m = mv[mv.club_id == cid]
        a, b = st.columns(2)
        html(T.moves("Arrivals", [(r.player_name, r.other_club, r.fee_m) for r in m[m.kind == "in"].itertuples()], T.ACCENT), a)
        html(T.moves("Departures", [(r.player_name, r.other_club, r.fee_m) for r in m[m.kind == "out"].itertuples()],
                     "#FF6B5E"), b)

# ================================================================== 02 signings 2026
html(T.section("02", "signings", "Summer 2026 signings", "Not played yet",
               "Every EUR 5M+ signing into a 2026/27 top-five league club in our data (the window to early July). For each: "
               "the chance he becomes a regular, the league points he adds after the player he pushes out, and what the fee claims."))
if now is None:
    st.info("No summer 2026 signings yet. Run `python scripts/forecast_2026.py` to create them.")
else:
    c1, c2, c3 = st.columns(3)
    leagues = c1.multiselect("League", list(D.LEAGUE_NAMES), default=[], format_func=lambda x: D.LEAGUE_NAMES.get(x, x),
                             key="now_league")
    verdicts = c2.multiselect("Verdict", D.VERDICT_CHOICES, default=[], key="now_verdict")
    text = c3.text_input("Search player or club", value="", key="now_text")
    sub = D.filter_signings(now, None, leagues, verdicts, 0.0, text)
    st.caption(f"{len(sub):,} signings · the 8 biggest fees as cards")
    if len(sub) == 0:
        st.warning("No signings match these filters.")
    else:
        cols = st.columns(4)
        for i, r in enumerate(sub.head(8).itertuples(index=False)):
            html(T.player_card(r), cols[i % 4])
        with st.expander(f"Full table ({len(sub):,} signings)"):
            tb = D.display_table(sub.fillna({"points_mean": 0, "points_p05": 0, "points_p95": 0, "claimed_points": 0,
                                             "confidence": 0}))
            tb = tb.drop(columns=["Season"]).astype(str)
            st.dataframe(tb, height=320, hide_index=True)
        html("<div class='fa-label' style='margin:26px 0 8px'>Scouting report</div>")
        pick = st.selectbox("Open one signing", list(sub.label), index=0, key="now_pick")
        scouting_report(sub[sub.label == pick].iloc[0], past=False)

# ================================================================== 03 transfer lab
html(T.section("03", "lab", "Transfer Lab", "Any player · any club · any fee",
               "Should my club sign him? Pick a 2026/27 club and a rated player. The lab plays him in the club's current "
               "squad, takes his minutes from the weakest players first, and turns the change in strength into league points."))
L = lab()
if L is None:
    st.info("Run `python scripts/forecast_2026.py` to create the Transfer Lab tables.")
else:
    clubs = fc[(fc.promoted == 0)] if fc is not None else None
    c1, c2 = st.columns(2)
    lg_lab = c1.selectbox("League", list(D.LEAGUE_NAMES), format_func=lambda x: D.LEAGUE_NAMES.get(x, x), key="lab_league")
    opts = clubs[clubs.league == lg_lab].sort_values("pred_rank") if clubs is not None else None
    club_name = c2.selectbox("Buying club", list(opts.club_name), key="lab_club")
    club_id = int(opts[opts.club_name == club_name].club_id.iloc[0])
    pl = L["players"][L["players"].club_id != club_id]
    who = st.selectbox(f"Player ({len(pl):,} rated players, best rated first; type to search)", list(pl.label), key="lab_player")
    p = pl[pl.label == who].iloc[0]
    c1, c2, c3 = st.columns(3)
    default_fee = float(round(p.value_m, 0)) if "value_m" in p and p.value_m == p.value_m and p.value_m > 0 else 30.0
    fee = c1.number_input("Fee (EUR M)", min_value=0.0, max_value=300.0, value=max(default_fee, 1.0), key="lab_fee")
    share = c2.slider("His share of the league minutes", min_value=0.1, max_value=1.0, value=0.7, step=0.05, key="lab_share",
                      help="About 0.7-0.9 for a regular starter, 0.4 for rotation, 0.15 for a squad player.")
    price = c3.selectbox("Price of a league point", ["Conservative: 1.3 pts per EUR 100M", "Between clubs: 6.7 pts per EUR 100M"],
                         key="lab_price")
    ppp = float(meta.get("price_low", 1.268)) if price.startswith("Cons") else float(meta.get("price_high", 6.7))
    res = D.lab_estimate(L, club_id, int(p.player_id), share, fee, ppp)
    shown = D.shown_verdict(res["verdict"], res["confidence"])
    a, b = st.columns([2, 3], gap="large")
    html(T.glass(f"<div class='fa-label'>{T.esc(p.player_name)} to {T.esc(club_name)}</div>"
                 f"<div class='fa-big' style='color:{T.ACCENT}'>{res['points_mean']:+.1f} pts</div>"
                 f"<div class='fa-sub'>a season · 90% range {res['points_p05']:+.1f} to {res['points_p95']:+.1f} · "
                 f"the fee claims {res['claimed_points']:.1f}</div>"
                 f"<div style='margin:16px 0 6px'>{T.badge(shown)}</div>"
                 f"<div class='fa-sub'>{100 * res['confidence']:.0f}% for the most likely outcome</div>"
                 + T.prob_bar(res["p_overpaid"], res["p_fair"], res["p_best_buy"])), a)
    lost = "".join(f"<div class='fa-move'><span><b>{T.esc(n)}</b> <span style='color:{T.MUTED}'>· rating {r:+.2f}</span></span>"
                   f"<span>{100 * s:.0f}% of the minutes</span></div>" for n, s, r in res["displaced"][:6])
    html(T.glass(f"<div class='fa-label' style='margin-bottom:6px'>Who loses minutes (weakest first)</div>{lost}"
                 f"<div class='fa-sub' style='margin-top:12px'>His rating {res['rating']:+.2f} goal difference per 90 "
                 f"(± {p.rating_sd:.2f}). Ratings come from matches up to May 2026.</div>"), b)
    st.caption("Model estimate, not a validated number: points added by one player could not be confirmed by an independent "
               "test (see Track record). Use it to compare options, not as a fact.")
    with st.expander("Your own numbers (verdict calculator)"):
        l2, r2 = st.columns(2, gap="large")
        fee2 = l2.number_input("Fee (EUR M)", min_value=1.0, max_value=300.0, value=40.0, key="calc_fee")
        gain = l2.number_input("Expected extra points per season", min_value=-10.0, max_value=15.0, value=2.0, key="calc_gain")
        sd_ = l2.slider("Uncertainty (standard deviation, points)", min_value=0.2, max_value=5.0, value=1.8, key="calc_sd")
        price2 = l2.number_input("Points per EUR 100M (1.3 within clubs, 6.7 between clubs)", min_value=0.5, max_value=15.0,
                                 value=1.3, key="calc_price")
        tol = l2.slider("'Fair' tolerance (points)", min_value=0.0, max_value=3.0, value=1.0, key="calc_tol")
        cr = D.calculator(fee2, gain, sd_, price2, tol)
        html(T.glass(f"<div class='fa-label'>Verdict</div><div style='margin:14px 0'>"
                     f"{T.badge(D.shown_verdict(cr['verdict'], cr['confidence']))}</div>"
                     f"<div class='fa-sub'>{100 * cr['confidence']:.0f}% for the most likely outcome · the fee claims "
                     f"{cr['claimed_points']:.2f} pts</div>" + T.prob_bar(cr["p_overpaid"], cr["p_fair"], cr["p_best_buy"])), r2)

# ================================================================== 04 track record
html(T.section("04", "evidence", "Track record", "Past seasons, predicted blind",
               "Why trust the forecast? We ran the same method on every season from 2019/20 to 2025/26, each time using only "
               "what was known before that season, then compared it with what really happened. This is the test, not the product."))
if "season_accuracy" in h:
    a, b, c, d_ = st.columns(4)
    html(T.stat_card("Season: top or bottom half", pct(h["season_accuracy"]),
                     f"95% CI {pct(h['season_lo'])} to {pct(h['season_hi'])}", accent=True), a)
    if "points_mae" in h:
        html(T.stat_card("Season points error", f"{h['points_mae']:.1f}", f"vs {h['points_mae_last']:.1f} for last season's points"), b)
    if "m8_accuracy" in h:
        html(T.stat_card("Signing becomes a regular", pct(h["m8_accuracy"]), f"{pct(h['m8_conf_accuracy'])} when confident"), c)
    html(T.stat_card("Coin flip", "50.0%", "chance"), d_)
a, b = st.columns(2)
val = []
if "season_accuracy" in h:
    val.append(f"Season top/bottom half: {pct(h['season_accuracy'])} (chance 50%), tied with strong baselines")
if "points_mae" in h:
    val.append(f"Season points error {h['points_mae']:.1f} vs {h['points_mae_last']:.1f} for last season's points")
if "m8_accuracy" in h:
    val.append(f"Signing becomes a regular: {pct(h['m8_accuracy'])} overall, {pct(h['m8_conf_accuracy'])} when confident")
if "rq1_hybrid" in h:
    val.append(f"Match model beats Elo: {h['rq1_hybrid']:.4f} vs {h['rq1_elo']:.4f} log loss")
html(T.evidence("Validated on real outcomes", val, T.ACCENT), a)
html(T.evidence("Not validated yet", [
    "Points added by a single signing (the independent on/off check finds no detectable link; it has low power)",
    "Fee verdicts (they rest on those points and on an uncertain price of a point)",
    "Fixed 30 Oct 2026: verdicts were biased against big clubs (46% of their signings 'Overpaid'); now 6%"], T.AMBER), b)

sd = season()
if sd is not None:
    html("<div class='fa-label' style='margin:30px 0 10px'>Past seasons, predicted blind · our pre-season table next to the real one</div>")
    c1, c2 = st.columns(2)
    yr = c1.selectbox("Season", sorted(sd.season.unique(), reverse=True), format_func=D.season_label, key="tr_season")
    lgp = c2.selectbox("League", sorted(sd.league.unique()), format_func=lambda x: D.LEAGUE_NAMES.get(x, x), key="tr_league")
    tbl = D.season_table(sd, yr, lgp)
    right_share = (tbl["Right?"] == "Yes").mean()
    html(f"<div class='fa-label' style='margin:14px 0 10px'>{T.esc(D.LEAGUE_NAMES.get(lgp, lgp))} {D.season_label(yr)} · "
         f"<span style='color:{T.ACCENT}'>{100 * right_share:.0f}% in the right half</span></div>")
    html(T.season_table(tbl))

df = signings()
html("<div class='fa-label' style='margin:30px 0 10px'>Past signings · what the model would have said on transfer day, "
     "next to what really happened</div>")
c1, c2, c3, c4 = st.columns(4)
seasons = c1.multiselect("Season", sorted(df.season.unique(), reverse=True), default=[], format_func=D.season_label, key="ps_season")
leagues = c2.multiselect("League", sorted(df.to_league.unique()), default=[], format_func=lambda x: D.LEAGUE_NAMES.get(x, x),
                         key="ps_league")
verdicts = c3.multiselect("Verdict", D.VERDICT_CHOICES, default=[], key="ps_verdict")
min_fee = c4.slider("Minimum fee (EUR M)", min_value=5, max_value=100, value=5, key="ps_fee")
text = st.text_input("Search player or club", value="", key="ps_text")
sub = D.filter_signings(df, seasons, leagues, verdicts, float(min_fee), text)
st.caption(f"{len(sub):,} past signings (2019-2025)")
if len(sub) == 0:
    st.warning("No signings match these filters.")
else:
    with st.expander(f"Full table ({len(sub):,} signings)"):
        tbl = D.display_table(sub)
        st.dataframe(tbl.astype(str), height=320, hide_index=True)
        st.download_button("Download this table (CSV)", tbl.to_csv(index=False), file_name="freeagent_past_signings.csv")
    pick = st.selectbox("Open one past signing", list(sub.label), index=0, key="ps_pick")
    scouting_report(sub[sub.label == pick].iloc[0], past=True)

t0, t1, t2, t3, t4, t5 = st.tabs(["Season prediction", "Will he become a regular?", "Match prediction", "Rating prior",
                                  "Reality check", "Bias fix"])
with t0:
    ss = report("season_summary.csv", 0)
    if ss is None:
        st.info("Run `python scripts/season_predict.py` to create the season results.")
    else:
        show = ss[["accuracy", "ci_lo", "ci_hi", "ours_minus_this", "verdict", "n"]].copy()
        show.columns = ["Accuracy", "95% CI low", "95% CI high", "Ours minus this", "Verdict", "Club-seasons"]
        show = show.round(3).astype(object).where(show.notna(), "").astype(str)   # text only: no mixed types
        show.index.name = "Method"
        st.dataframe(show)
        st.caption("Will a club finish in the top half? Predicted before the season, nothing tuned. Every method "
                   "predicts exactly half of each league as top half.")
    pt = report("season_points_test.csv")
    if pt is not None:
        names = {"measure": "Measure", "ours": "Ours", "last_season": "Same as last season",
                 "ours_advantage": "Ours better by", "league_season_ci_lo": "95% CI low",
                 "league_season_ci_hi": "95% CI high", "league_season_verdict": "Verdict"}
        cols = [c for c in names if c in pt]
        st.markdown("**Points forecasts against 'same points as last season'** (clubs that stayed in the league)")
        st.dataframe(pt[cols].round(3).rename(columns=names).astype(str), hide_index=True)
        st.caption("Bootstrap over whole league-seasons, the stricter test, since clubs in one table are not independent.")
with t1:
    m8 = report("m8_summary.csv", 0)
    if m8 is not None:
        show = m8[["accuracy", "auc", "log_loss", "confident_accuracy", "coverage", "n"]].round(3)
        show.columns = ["Accuracy", "AUC", "Log loss", "Accuracy when confident", "Share confident", "Signings"]
        show = show.astype(object)
        show.loc[show["Share confident"] == 0, ["Accuracy when confident", "Share confident"]] = "n/a (never confident)"
        show = show.astype(str)                        # text only: no mixed types
        st.dataframe(show)
        st.caption("Ours: logistic regression and gradient boosting averaged. 'Confident' = 60% or more, or 40% or less.")
    bs = report("m8_by_season.csv")
    if bs is not None:
        ours = bs[bs.model == "Ensemble: logistic + boosting (ours)"].set_index("season")
        html(T.hbars({D.season_label(s): 100 * v for s, v in ours.accuracy.items()}))
with t2:
    rq = report("rq1_summary.csv", 0)
    if rq is not None:
        st.dataframe(rq.round(4).astype(str))
        st.caption("Lower log loss is better. Player ratings + Elo beat Elo alone by a small but significant margin. "
                   "About 52% accuracy is normal for football matches.")
with t3:
    m2 = report("m2_match_models.csv", 0)
    if m2 is not None:
        st.dataframe(m2.round(4).astype(str))
        st.caption("The machine-learning starting guess improves the player model on average, not in every year.")
with t4:
    m7 = report("m7_validation.csv")
    if m7 is not None:
        st.dataframe(m7.round(3).astype(str))
    st.warning("Predicted points vs how each club's goal difference changed with the player on versus off the pitch: "
               "no detectable link, but the test is weak (about 34% power). The points estimates are model outputs "
               "with uncertainty, not confirmed facts.")
with t5:
    st.markdown(
        "**What was wrong.** A signing's value is how much better he is than the player who loses his place. The first "
        "version (v1) compared him with the *average* player in his position, and used the squad from May, before the "
        "summer departures. At a big club the average incumbent is a star, so a world-class signing looked like no upgrade: "
        "46% of the signings by the strongest quarter of clubs were called 'Overpaid', against 2% for the weakest quarter "
        "(Bellingham, Rice, Kane, Grealish...).\n\n"
        "**The fix (v2, one change of assumption, nothing tuned).** Managers drop their weakest players first, so a "
        "signing now takes his minutes from the lowest-rated players in his position, and the squad is the summer-aware "
        "squad the season forecast uses. Result: 'Overpaid' in the strongest quarter fell from 46% to 6%; the reality "
        "check (on/off goal difference) is still inconclusive in both versions, so verdicts stay labelled as not validated.")
    v1 = report("m5_signing_points_v1.csv")
    if v1 is not None:
        k = ["player_id", "season", "to_club_id"]
        m = v1[k + ["player_name", "to_club_name", "verdict", "points_mean"]].merge(
            df[k + ["verdict", "points_mean"]], on=k, suffixes=(" v1", " v2"))
        star = m[m.player_name.isin(["Jude Bellingham", "Harry Kane", "Declan Rice", "Jack Grealish", "Josko Gvardiol",
                                     "Kai Havertz", "Randal Kolo Muani", "Antoine Griezmann", "Romelu Lukaku"])]
        star = star.rename(columns={"player_name": "Player", "to_club_name": "Club", "season": "Season"})
        st.dataframe(star.drop(columns=["player_id", "to_club_id"]).round(2).astype(str), hide_index=True)

# ================================================================== 05 method
html(T.section("05", "method", "How it works", "Eight steps",
               "From one match to a whole season: every step uses only information available at the time."))
steps = [
    ("Rate every player", "Bayesian plus-minus: for every match we ask how much goal difference each player's minutes "
     "explain, across five leagues and twelve seasons. Every rating comes with an uncertainty."),
    ("Starting guess from machine learning", "Gradient boosting predicts a rating from goals, assists, cards, minutes, "
     "age and position. It never sees market value or team results."),
    ("Predict matches", "Goals follow a Poisson distribution whose rate depends on the gap in team strength "
     "(Dixon-Coles style). Added to Elo it beats Elo alone."),
    ("Value a signing", "His club gets stronger by his minutes times (his rating minus the rating of the weakest players "
     "he pushes out). We play the league season with and without him."),
    ("Price the points", "Squad value is linked to league points: about 1.3 points per EUR 100M within clubs, 6.7 "
     "between clubs. We use the conservative 1.3 and show 6.7 as a sensitivity."),
    ("Will he become a regular?", "A classifier (logistic regression plus boosting) predicts from pre-transfer "
     "information whether he plays at least half of the league minutes in his first season."),
    ("Test on the past", "Steps 1-6 run blind on 2019-2025: each season predicted with models trained only on earlier "
     "seasons, then checked against the real final tables and minutes."),
    ("Forecast the live season", "The same frozen settings on all data up to the summer of 2026: the 2026/27 table, the "
     "summer's signings and the Transfer Lab."),
]
s1, s2 = st.columns(2)
for i, (title, txt) in enumerate(steps, 1):
    html(T.step(i, title, txt), s1 if i % 2 else s2)
with st.expander("Limitations (read these)"):
    st.markdown("""
- The 2026/27 forecast cannot be scored until May 2027; our accuracy claims come from the 2019-2025 track record.
- Our data ends in early July 2026, so late-summer transfers (August deadline day) are missing from the forecast.
- Promoted clubs have no second-division matches in our data; they get the average strength of the clubs they replaced,
  and their signings are not judged.
- Points estimates and verdicts are not confirmed by an independent test (low statistical power).
- Individual ratings are uncertain; players new to the top five leagues have no rating and are shown as 'Not rated'.
- The price of a league point is not well identified: 1.3 (within club) versus 6.7 (between clubs) per EUR 100M.
- Season prediction matches, but does not beat, simple baselines on top or bottom half; its points forecasts are closer.
- Transfermarkt data are scraped; line-up coverage is about 99% in most seasons but lower in a few.
""")
html(T.footer("FreeAgent · Data: Transfermarkt via Kaggle · every number comes from code in the repository"))
