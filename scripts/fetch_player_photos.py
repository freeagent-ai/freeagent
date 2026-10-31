"""Find a FREELY LICENSED photo for each signed player on Wikimedia Commons (via Wikidata), with its credit.

Why Wikimedia Commons: it only hosts freely licensed or public-domain files, and every file lists its author
and licence. We store that credit and the app shows it next to the photo (most licences require attribution).
Matching: Wikidata search by name, keep only association football players (P106 = Q937857), and require the
same date of birth as in our data when both are known. If nothing matches safely, the player keeps his
initials avatar. Nothing is downloaded: the app links to the Commons thumbnail.

Needs internet. Resumable: re-running skips players already looked up.
Usage: python scripts/fetch_player_photos.py            (all 1,592 players, biggest fees first; ~10-20 min)
       python scripts/fetch_player_photos.py --limit 200
Output: reports/player_photos.csv  (player_id, wikidata, file, thumb_url, page_url, artist, license, license_url)
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freeagent import config

UA = {"User-Agent": "FreeAgent-student-project/1.0 (educational capstone; looks up freely licensed player photos)"}
FOOTBALLER = "Q937857"
OUT = config.REPORTS / "player_photos.csv"
COLS = ["player_id", "wikidata", "file", "thumb_url", "page_url", "artist", "license", "license_url"]


def get(url: str, params: dict) -> dict:
    q = url + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}


def _claim_values(ent: dict, prop: str) -> list:
    out = []
    for c in ent.get("claims", {}).get(prop, []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if v is not None:
            out.append(v)
    return out


def choose_entity(entities: dict, dob: str | None) -> tuple[str, str] | None:
    """Pick (qid, image file) among candidate Wikidata entities. Must be a footballer with an image; if our date
    of birth is known it must match exactly; if it is unknown, only accept a single footballer candidate."""
    ok = []
    for qid, ent in entities.items():
        occ = {v.get("id") for v in _claim_values(ent, "P106") if isinstance(v, dict)}
        imgs = [v for v in _claim_values(ent, "P18") if isinstance(v, str)]
        if FOOTBALLER not in occ or not imgs:
            continue
        births = [v.get("time", "")[1:11] for v in _claim_values(ent, "P569") if isinstance(v, dict)]
        if dob:
            if dob in births:
                return qid, imgs[0]
        else:
            ok.append((qid, imgs[0]))
    return ok[0] if len(ok) == 1 else None


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_info(files: list[str]) -> dict:
    res = get("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 320,
        "titles": "|".join("File:" + f for f in files)})
    out = {}
    for page in res.get("query", {}).get("pages", {}).values():
        ii = (page.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        name = page.get("title", "").replace("File:", "", 1)
        out[name] = {"thumb_url": ii.get("thumburl"), "page_url": ii.get("descriptionurl"),
                     "artist": strip_html(md.get("Artist", {}).get("value", ""))[:120],
                     "license": md.get("LicenseShortName", {}).get("value", ""),
                     "license_url": md.get("LicenseUrl", {}).get("value", "")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from freeagent import data
    cols = ["player_id", "player_name", "transfer_fee"]
    parts = [pd.read_csv(config.REPORTS / f, usecols=cols).assign(now=int(f.startswith("forecast")))
             for f in ("forecast_2026_signings.csv", "m5_signing_points.csv") if (config.REPORTS / f).exists()]
    sig = pd.concat(parts)          # summer 2026 signings first (they are on the front page), then the rest by fee
    sig = sig.sort_values(["now", "transfer_fee"], ascending=False).drop_duplicates("player_id")
    players = data.load_tables(config.DATA / "freeagent_subset.zip")["players"].set_index("player_id")
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    todo = sig[~sig.player_id.isin(done.player_id)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already looked up, {len(todo)} to go")
    rows, t0 = [], time.time()
    for i, r in enumerate(todo.itertuples(index=False), 1):
        dob = players.date_of_birth.get(r.player_id)
        dob = pd.Timestamp(dob).strftime("%Y-%m-%d") if pd.notna(dob) else None
        hits = get("https://www.wikidata.org/w/api.php", {"action": "wbsearchentities", "search": r.player_name,
                                                          "language": "en", "type": "item", "limit": 7})
        ids = [h["id"] for h in hits.get("search", [])]
        pick = None
        if ids:
            ents = get("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(ids),
                                                              "props": "claims"}).get("entities", {})
            pick = choose_entity(ents, dob)
        rows.append({"player_id": r.player_id, "wikidata": pick[0] if pick else "", "file": pick[1] if pick else ""})
        time.sleep(0.1)
        if i % 50 == 0 or i == len(todo):
            new = pd.DataFrame(rows)
            files = [f for f in new.file if f]
            info = {}
            for k in range(0, len(files), 40):
                info.update(commons_info(files[k:k + 40]))
            for c in COLS[3:]:
                new[c] = [info.get(f.replace("_", " "), info.get(f, {})).get(c, "") if f else "" for f in new.file]
            done = pd.concat([done, new[COLS]], ignore_index=True)
            done.to_csv(OUT, index=False)
            rows = []
            found = (done.thumb_url.fillna("") != "").sum()
            print(f"  {len(done)} looked up, {found} photos found  [{time.time() - t0:.0f}s]", flush=True)
    found = done[done.thumb_url.fillna("") != ""]
    print(f"\nDone. {len(found)} of {len(done)} players have a freely licensed photo. Saved {OUT.name}")
    if len(found):
        print("Licences:", found.license.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
