"""Resolve the hand-collected late-window 2026 deals (data/supplement/late_window_2026_raw.csv) to our player and club ids.

Why: the Transfermarkt snapshot stops in early July 2026 but the window ran to 1 September, so big deals
(Diomande, Man City's Fernandez/Anderson/Bouaddi, Barcelona's Rodri, ...) are missing. The fees are PRESS-REPORTED
(ESPN, Sportsmole, LaLiga/beIN, worldsoccertalk, Flashscore), pounds converted at GBP_EUR; they are not Transfermarkt data.
Output: data/supplement/late_window_2026.csv (+ a list of rows that could not be matched).
"""
import sys, unicodedata, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GBP_EUR = 1.15
WINDOW_DATE = "2026-08-20"
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()
ALIAS = {"coventry city": "coventry", "inter": "inter milan", "ac milan": "ac milan", "atletico madrid": "atletico de madrid", "tottenham hotspur": "tottenham hotspur",
         "bayer 04 leverkusen": "bayer 04 leverkusen", "fc barcelona": "fc barcelona", "as roma": "as roma", "as monaco": "as monaco"}

z = zipfile.ZipFile(ROOT / "data" / "freeagent_subset.zip")
rd = lambda n: pd.read_csv(z.open(n))
pl, cl, tr = rd("players.csv"), rd("clubs.csv"), rd("transfers.csv")
names = pd.concat([cl[["club_id", "name"]].rename(columns={"name": "n"}),
                   tr[["to_club_id", "to_club_name"]].set_axis(["club_id", "n"], axis=1),
                   tr[["from_club_id", "from_club_name"]].set_axis(["club_id", "n"], axis=1)]).dropna().drop_duplicates()
names["k"] = names.n.map(norm)
pl["k"] = pl["name"].map(norm)
pl["club_k"] = pl.current_club_name.map(norm)

TAB = pd.read_csv(ROOT / "reports" / "forecast_2026_table.csv")[["club_id", "club_name"]]
TAB["k"] = TAB.club_name.map(norm)
HINT = {"leipzig": "rb leipzig", "nottingham": "nottingham forest", "paris": "paris saint-germain", "tottenham": "tottenham hotspur",
        "brighton": "brighton & hove albion", "west ham": "west ham", "milan": "ac milan", "bayern": "bayern munich",
        "bologna": "bologna football club 1909", "sporting": "sporting cp", "wolverhampton": "wolverhampton wanderers"}
def club_id(txt, table_only=False):
    k = ALIAS.get(norm(txt), norm(txt)); k = HINT.get(k, k)
    m = TAB[TAB.k == k]
    if m.empty:
        m = TAB[TAB.k.str.contains(k, regex=False)]
    if len(m):
        return int(m.club_id.iloc[0]), m.club_name.iloc[0]
    if table_only:
        return None, None
    m = names[names.k == k]
    if m.empty:
        m = names[names.k.str.contains(k, regex=False)]
    return (int(m.club_id.iloc[0]), m.n.iloc[0]) if len(m) else (None, None)

raw = pd.read_csv(ROOT / "data" / "supplement" / "late_window_2026_raw.csv")
rows, bad = [], []
for r in raw.itertuples():
    k = norm(r.player_name)
    c = pl[(pl.k == k) & (pl.last_season >= 2025)]
    if c.empty:
        c = pl[(pl.k == k)]
    if len(c) > 1:                                   # tie-break: current club matches the "from" hint, else most valuable
        h = c[c.club_k.str.contains(norm(r.from_hint), regex=False)]
        c = h if len(h) else c.sort_values("market_value_in_eur", ascending=False)
    cid, cname = club_id(r.to_club)
    if c.empty or cid is None:
        bad.append((r.player_name, r.to_club, "player" if c.empty else "club")); continue
    p = c.iloc[0]
    fid, fname = club_id(r.from_hint, table_only=True)
    if int(p.current_club_id) == cid:             # already at the buyer in our data: not a late deal
        continue
    if fid is None:
        fid, fname = p.current_club_id, p.current_club_name
    fee = r.fee * (GBP_EUR if r.ccy == "GBP" else 1.0) * 1e6
    rows.append({"player_id": int(p.player_id), "player_name": p["name"], "transfer_date": WINDOW_DATE,
                 "transfer_season": "26/27", "from_club_id": fid, "to_club_id": cid,
                 "from_club_name": fname, "to_club_name": cname, "transfer_fee": fee,
                 "market_value_in_eur": p.market_value_in_eur, "source": r.source, "fee_ccy": r.ccy})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "supplement" / "late_window_2026.csv", index=False)
print(len(out), "resolved,", len(bad), "unresolved")
pd.set_option("display.width", 250)
print(out[["player_name", "from_club_name", "to_club_name", "transfer_fee"]].assign(fee_m=lambda d: (d.transfer_fee / 1e6).round(1)).drop(columns="transfer_fee").to_string())
print("UNRESOLVED:", bad)
