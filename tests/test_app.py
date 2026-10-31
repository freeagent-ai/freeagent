"""Tests for the app helpers, and a dry run of app.py with a stand-in for Streamlit.

The dry run executes every page of app.py top to bottom, so a typo or a wrong column name
fails here instead of in front of the examiner. (Real Streamlit is not needed for this test.)
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeagent import app_data as D


def _df():
    return D.load_signings()


def test_load_and_filter_signings():
    df = _df()
    assert len(df) > 1000 and {"label", "fee_m", "season_label", "verdict"} <= set(df.columns)
    sub = D.filter_signings(df, seasons=[2023], leagues=["GB1"], verdicts=["Overpaid"], min_fee=20.0)
    assert (sub.season == 2023).all() and (sub.to_league == "GB1").all() and (sub.fee_m >= 20).all()
    assert D.season_label(2023) == "2023/24"


def test_display_table_and_sentences():
    r = _df().iloc[0]
    assert "Verdict" in D.display_table(_df().head(3)).columns
    assert "played" in D.what_happened(r) and "points" in D.verdict_sentence(r)


def test_low_confidence_is_never_called():
    assert D.shown_verdict("Overpaid", 0.34) == "Too close to call"
    assert D.shown_verdict("Overpaid", 0.74) == "Overpaid"
    df = _df()
    rated = df[df.known_player == 1]                      # unrated players show "Not rated" instead (tested below)
    assert (rated.shown_verdict[rated.confidence < D.CONFIDENT_MIN] == "Too close to call").all()
    row = rated[rated.confidence < D.CONFIDENT_MIN].iloc[0]
    assert "too close to call" in D.verdict_sentence(row)


def test_unrated_players_are_not_judged():
    assert D.shown_verdict("Fair", 0.9, known=False) == "Not rated"
    df = _df()
    assert (df.shown_verdict[df.known_player == 0] == "Not rated").all()
    row = df[df.known_player == 0].iloc[0]
    assert "do not judge" in D.verdict_sentence(row)


def test_calculator_matches_pipeline_logic():
    big = D.calculator(100, 0.0, 1.0, 7.0, 1.0)      # claims 7 points (high price), adds 0
    star = D.calculator(10, 6.0, 1.0, 7.0, 1.0)      # claims 0.7, adds 6
    assert big["verdict"] == "Overpaid" and star["verdict"] == "Best Buy"
    assert abs(big["p_overpaid"] + big["p_fair"] + big["p_best_buy"] - 1) < 1e-9


def test_headline_numbers_are_read_from_reports():
    h = D.headline_numbers()
    assert 0.55 < h["m8_accuracy"] < 0.75 and h["rq1_hybrid"] < h["rq1_elo"]


def _panel(page):
    """A stand-in for st / a column / a tab: widgets return their default values, like the real thing."""
    p = MagicMock()
    p.columns.side_effect = lambda spec, **kw: [_panel(page) for _ in range(spec if isinstance(spec, int) else len(spec))]
    p.tabs.side_effect = lambda labels: [_panel(page) for _ in labels]
    p.radio.side_effect = lambda label, options, **kw: page
    p.selectbox.side_effect = lambda label, options, index=0, **kw: list(options)[index]
    p.multiselect.side_effect = lambda label, options, default=None, **kw: default or []
    p.slider.side_effect = lambda label, **kw: kw["value"]
    p.number_input.side_effect = lambda label, **kw: kw["value"]
    p.text_input.side_effect = lambda label, value="", **kw: value
    return p


def _fake_streamlit(page):
    st = _panel(page)
    st.sidebar = _panel(page)
    st.cache_data = lambda f=None, **kw: (f if f is not None else (lambda g: g))
    st.cache_resource = st.cache_data
    return st


def _ensure_season_reports():
    """The season files exist on the machine that ran season_predict.py. If they are missing here, make
    synthetic stand-ins (removed again afterwards) so every page is still exercised."""
    import shutil
    import subprocess
    made = []
    rep = ROOT / "reports"
    if not (rep / "season_predictions.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "season_predict.py"), "--synthetic"], check=True,
                       capture_output=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "season_points_test.py"), "--boot", "100",
                        "--file", str(rep / "season_synthetic_predictions.csv")], check=True, capture_output=True)
        for src, dst in [("season_synthetic_predictions.csv", "season_predictions.csv"),
                         ("season_synthetic_summary.csv", "season_summary.csv"),
                         ("season_points_test_synthetic.csv", "season_points_test.csv")]:
            shutil.copy(rep / src, rep / dst)
            made.append(rep / dst)
        for f in rep.glob("*synthetic*.csv"):
            f.unlink()
    return made


def test_season_helpers():
    made = _ensure_season_reports()
    try:
        sd = D.load_season()
        assert sd is not None and {"club_name", "league_name", "correct"} <= set(sd.columns)
        r = sd.iloc[0]
        tbl = D.season_table(sd, r.season, r.league)
        assert len(tbl) == (sd[(sd.season == r.season) & (sd.league == r.league)]).shape[0]
        assert set(tbl["Right?"]) <= {"Yes", "No"}
        h = D.headline_numbers()
        assert 0.5 < h["season_accuracy"] <= 1.0
    finally:
        for f in made:
            f.unlink()


def test_every_page_of_the_app_runs():
    made = _ensure_season_reports()
    try:
        _run_every_page()
    finally:
        for f in made:
            f.unlink()


def _run_every_page():
    code = (ROOT / "app.py").read_text()
    for page in ["single scrolling page"]:          # the app is one page; the loop is kept for future pages
        saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = _fake_streamlit(page)
        try:
            exec(compile(code, "app.py", "exec"), {"__name__": "__main__", "__file__": str(ROOT / "app.py")})
        finally:
            if saved is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = saved
