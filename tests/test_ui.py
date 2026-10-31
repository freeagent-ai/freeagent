"""Tests for the app's visual extras: flag codes, the photo matcher (offline), and card HTML."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeagent import app_data as D, flags, theme as T

spec = importlib.util.spec_from_file_location("fetch_player_photos", ROOT / "scripts" / "fetch_player_photos.py")
FP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FP)


def _ent(occ, dob, img="Player.jpg"):
    claims = {"P106": [{"mainsnak": {"datavalue": {"value": {"id": o}}}} for o in occ],
              "P569": [{"mainsnak": {"datavalue": {"value": {"time": f"+{dob}T00:00:00Z"}}}}] if dob else []}
    if img:
        claims["P18"] = [{"mainsnak": {"datavalue": {"value": img}}}]
    return {"claims": claims}


def test_photo_matcher_requires_footballer_and_same_birthday():
    ents = {"Q1": _ent(["Q33999"], "1999-09-21", "Actor.jpg"),            # same name, an actor
            "Q2": _ent(["Q937857"], "1990-01-01", "Other.jpg"),            # footballer, wrong birthday
            "Q3": _ent(["Q937857"], "1999-09-21", "Right.jpg")}
    assert FP.choose_entity(ents, "1999-09-21") == ("Q3", "Right.jpg")
    assert FP.choose_entity({"Q2": ents["Q2"]}, "1999-09-21") is None     # never guess a different person
    assert FP.choose_entity({"Q2": ents["Q2"], "Q3": ents["Q3"]}, None) is None   # ambiguous without a birthday
    assert FP.choose_entity({"Q4": _ent(["Q937857"], "1999-09-21", None)}, "1999-09-21") is None   # no image


def test_flag_codes():
    assert flags.code("England") == "gb-eng" and flags.code("Brazil") == "br" and flags.code(None) is None
    assert set(flags.LEAGUE_FLAG) == {"GB1", "ES1", "IT1", "L1", "FR1"}


def test_cards_fall_back_to_initials_and_escape_names():
    r = D.load_signings().iloc[0]
    html = T.player_card(r)
    assert "fa-card" in html and ("fa-photo" in html or "fa-avatar" in html)
    r2 = r.copy()
    r2["player_name"], r2["photo"], r2["flag"] = "O'Neil <b>", "", "gb-eng"
    h2 = T.spotlight_card(r2)
    assert "O&#x27;Neil &lt;b&gt;" in h2 and "gb-eng" in h2
    assert "flagcdn.com" in T.league_badges(D.LEAGUE_NAMES)
