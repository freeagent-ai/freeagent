"""Visual theme for the FreeAgent app: one stylesheet and small HTML components.

Design: near-black background with large soft colour glows and film grain, frosted-glass cards, a sticky pill
menu that scrolls to numbered sections, extra-wide display type, monospace labels, one neon-green accent.
Fonts (free Google Fonts, open licence): Unbounded (display), JetBrains Mono (labels), Manrope (body).
No player photos or club/brand logos are used.
Every component returns HTML with NO leading indentation (Markdown would turn indented lines into code blocks);
render with st.markdown(html, unsafe_allow_html=True).
"""
from __future__ import annotations
import html as _html

ACCENT = "#3DF5A0"
CYAN = "#38D5FF"
VIOLET = "#A78BFA"
AMBER = "#FFB86B"
TEXT = "#F2F4FA"
MUTED = "#9AA3BF"
VERDICT = {"Overpaid": "#FF6B5E", "Fair": "#9AA4BF", "Best Buy": "#3DF5A0",
           "Too close to call": "#7C86A8", "Not rated": "#56607E"}

GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/>"
         "</filter><rect width='100%' height='100%' filter='url(%23n)' opacity='.55'/></svg>")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea, button {{ font-family: 'Manrope', system-ui, sans-serif; }}
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(700px 520px at 6% 4%, rgba(61,245,160,.20), transparent 62%),
    radial-gradient(760px 620px at 100% 30%, rgba(124,58,237,.30), transparent 62%),
    radial-gradient(900px 600px at 45% 105%, rgba(20,184,166,.20), transparent 60%),
    #050508;
}}
[data-testid="stAppViewContainer"]::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: url("{GRAIN}"); opacity: .07; mix-blend-mode: overlay; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ background: transparent; display: none; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ position: relative; z-index: 1; max-width: 1220px; padding-top: 6.2rem; padding-bottom: 4rem; }}
h1, h2, h3 {{ font-family: 'Unbounded', sans-serif !important; color: {TEXT}; letter-spacing: -.01em; }}
a {{ text-decoration: none !important; }}

/* menu + scroll progress */
.fa-nav {{ position: fixed; top: 14px; left: 50%; transform: translateX(-50%); z-index: 1000; width: min(1180px, calc(100% - 32px));
  display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 10px 8px 18px;
  background: rgba(12,12,18,.55); border: 1px solid rgba(255,255,255,.10); border-radius: 999px;
  backdrop-filter: blur(22px) saturate(150%); -webkit-backdrop-filter: blur(22px) saturate(150%); box-shadow: 0 10px 40px rgba(0,0,0,.45); }}
.fa-nav .brand {{ display: flex; align-items: center; gap: 9px; font-family: 'Unbounded', sans-serif; font-weight: 800;
  font-size: .95rem; letter-spacing: .04em; color: {TEXT}; white-space: nowrap; }}
.fa-nav .links {{ display: flex; gap: 2px; flex-wrap: wrap; justify-content: center; }}
.fa-nav .links a {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
  color: {MUTED} !important; padding: 9px 13px; border-radius: 999px; transition: all .2s; }}
.fa-nav .links a:hover {{ color: #0A0A10 !important; background: {TEXT}; }}
.fa-nav .cta {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
  color: #06120C !important; background: {ACCENT}; padding: 10px 16px; border-radius: 999px; font-weight: 700; white-space: nowrap; }}
.fa-progress {{ position: fixed; top: 0; left: 0; height: 3px; width: 100%; z-index: 1001; transform-origin: 0 50%;
  background: linear-gradient(90deg, {ACCENT}, {CYAN}, {VIOLET}); animation: fa-grow linear; animation-timeline: scroll(nearest block); }}
@keyframes fa-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
[id] {{ scroll-margin-top: 96px; }}

/* hero */
.fa-meta {{ display: flex; align-items: center; gap: 14px; font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .14em; text-transform: uppercase; color: {MUTED}; margin-bottom: 26px; }}
.fa-meta .line {{ flex: 1; height: 1px; background: rgba(255,255,255,.12); }}
.fa-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {ACCENT}; box-shadow: 0 0 12px {ACCENT}; margin-right: 8px;
  animation: fa-pulse 2.2s ease-in-out infinite; }}
@keyframes fa-pulse {{ 50% {{ box-shadow: 0 0 0 6px rgba(61,245,160,.12), 0 0 18px {ACCENT}; }} }}
.fa-display {{ font-family: 'Unbounded', sans-serif; font-weight: 900; text-transform: uppercase; line-height: .92;
  font-size: clamp(2.3rem, 5.6vw, 5rem); letter-spacing: -.02em; color: {TEXT}; margin: 0; }}
.fa-display .acc {{ background: linear-gradient(90deg, {ACCENT} 0%, {CYAN} 35%, {ACCENT} 70%); background-size: 220% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: fa-shine 7s linear infinite; }}
@keyframes fa-shine {{ to {{ background-position: -220% 0; }} }}
.fa-display .outline {{ color: transparent; -webkit-text-stroke: 1.6px rgba(242,244,250,.85); }}
.fa-hero-svg {{ display: block; width: 100%; height: auto; filter: drop-shadow(0 0 28px rgba(61,245,160,.28)); transition: filter .3s; }}
.fa-hero-svg:hover {{ filter: drop-shadow(0 0 46px rgba(61,245,160,.55)); }}
.fa-hero-svg ::selection {{ background: {ACCENT}; color: #06120C; }}
.fa-lead {{ font-size: 1.1rem; line-height: 1.65; color: #C7CDE0; max-width: 560px; margin: 26px 0 22px; }}
.fa-btn {{ display: inline-flex; align-items: center; gap: 8px; padding: 13px 20px; border-radius: 999px; margin: 0 8px 8px 0;
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }}
.fa-btn.primary {{ background: {ACCENT}; color: #06120C !important; box-shadow: 0 8px 30px rgba(61,245,160,.30); }}
.fa-btn.ghost {{ border: 1px solid rgba(255,255,255,.18); color: {TEXT} !important; background: rgba(255,255,255,.04); }}

/* glass */
.fa-glass {{ position: relative; background: linear-gradient(160deg, rgba(255,255,255,.075), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.10); border-radius: 22px; padding: 22px 24px; margin-bottom: 16px;
  backdrop-filter: blur(20px) saturate(140%); -webkit-backdrop-filter: blur(20px) saturate(140%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.07), 0 20px 50px rgba(0,0,0,.35); transition: transform .25s, border-color .25s; }}
.fa-glass:hover {{ border-color: rgba(255,255,255,.18); }}
.fa-lift:hover {{ transform: translateY(-4px); }}
.fa-label {{ font-family: 'JetBrains Mono', monospace; font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; color: {MUTED}; }}
.fa-big {{ font-family: 'Unbounded', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1; color: {TEXT}; margin: 12px 0 10px; }}
.fa-sub {{ font-size: .9rem; color: {MUTED}; line-height: 1.5; }}
.fa-rows .r {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 13px 0;
  border-bottom: 1px solid rgba(255,255,255,.07); }}
.fa-rows .r:last-child {{ border-bottom: 0; }}
.fa-rows .v {{ font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: .92rem; letter-spacing: .04em; color: {TEXT}; text-align: right; }}

/* ticker */
.fa-ticker {{ overflow: hidden; border-top: 1px solid rgba(255,255,255,.08); border-bottom: 1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.025); margin: 34px -2rem 40px; padding: 16px 0; }}
.fa-ticker .t {{ display: flex; width: max-content; gap: 34px; animation: fa-marquee 38s linear infinite; }}
.fa-ticker span {{ font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: 1.15rem; text-transform: uppercase;
  color: rgba(242,244,250,.72); white-space: nowrap; }}
.fa-ticker b {{ color: {ACCENT}; font-size: .8rem; }}
@keyframes fa-marquee {{ to {{ transform: translateX(-50%); }} }}

/* section heads */
.fa-sec {{ margin: 56px 0 22px; }}
.fa-sec .top {{ display: flex; align-items: center; gap: 14px; font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .16em; text-transform: uppercase; color: {MUTED}; }}
.fa-sec .top .n {{ color: {ACCENT}; }}
.fa-sec .top .line {{ flex: 1; height: 1px; background: rgba(255,255,255,.12); }}
.fa-sec h2 {{ font-family: 'Unbounded', sans-serif !important; font-weight: 900 !important; text-transform: uppercase;
  font-size: clamp(2rem, 4.6vw, 3.6rem) !important; line-height: .98 !important; margin: 18px 0 10px !important; padding: 0 !important; }}
.fa-sec p {{ color: #C7CDE0; font-size: 1.02rem; max-width: 720px; margin: 0; }}

/* way-in cards */
.fa-way {{ min-height: 200px; display: flex; flex-direction: column; }}
.fa-way h3 {{ font-weight: 900 !important; text-transform: uppercase; font-size: 1.45rem !important; line-height: 1.05 !important; margin: 14px 0 10px !important; padding: 0 !important; }}
.fa-way p {{ color: {MUTED}; font-size: .92rem; line-height: 1.55; flex: 1; margin: 0 0 14px; }}
.fa-way .go {{ font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; }}

/* chips, badges, bars */
.fa-chip {{ display: inline-block; padding: 4px 11px; border-radius: 999px; font-family: 'JetBrains Mono', monospace; font-size: .66rem;
  letter-spacing: .1em; text-transform: uppercase; border: 1px solid rgba(255,255,255,.18); color: {TEXT}; }}
.fa-badge {{ display: inline-block; padding: 8px 18px; border-radius: 999px; font-family: 'Unbounded', sans-serif; font-weight: 800;
  font-size: 1rem; letter-spacing: .06em; text-transform: uppercase; }}
.fa-bar {{ display: flex; height: 36px; border-radius: 999px; overflow: hidden; gap: 3px; margin: 10px 0 4px; }}
.fa-bar div {{ display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .8rem; color: #07070C;
  min-width: 0; white-space: nowrap; overflow: hidden; }}

/* player cards */
.fa-card {{ position: relative; overflow: hidden; border-radius: 22px; padding: 18px; margin-bottom: 16px;
  background: linear-gradient(165deg, rgba(255,255,255,.09), rgba(255,255,255,.02)); border: 1px solid rgba(255,255,255,.10);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); transition: transform .25s, border-color .25s; }}
.fa-card:hover {{ transform: translateY(-4px); border-color: rgba(255,255,255,.26); box-shadow: 0 24px 60px rgba(61,245,160,.10); }}
.fa-card .glow {{ position: absolute; width: 160px; height: 160px; border-radius: 50%; right: -50px; top: -60px; filter: blur(30px); opacity: .35; }}
.fa-card .head {{ display: flex; align-items: center; gap: 12px; position: relative; }}
.fa-avatar {{ width: 48px; height: 48px; border-radius: 16px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  font-family: 'Unbounded', sans-serif; font-weight: 800; font-size: .92rem; color: #fff;
  background: linear-gradient(135deg, rgba(56,213,255,.9), rgba(167,139,250,.9)); }}
.fa-name {{ font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: .95rem; color: {TEXT}; line-height: 1.2; }}
.fa-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; margin-top: 14px; position: relative; }}
.fa-stat {{ background: rgba(255,255,255,.05); border-radius: 12px; padding: 9px 10px; }}
.fa-stat b {{ display: block; font-family: 'Unbounded', sans-serif; font-size: 1rem; color: {TEXT}; }}
.fa-stat span {{ font-family: 'JetBrains Mono', monospace; font-size: .6rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .1em; }}

/* tables and bars */
table.fa-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
table.fa-table th {{ text-align: left; font-family: 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase;
  font-size: .66rem; color: {MUTED}; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.12); white-space: nowrap; }}
table.fa-table td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,.06); color: {TEXT}; white-space: nowrap; }}
table.fa-table tr:hover td {{ background: rgba(255,255,255,.035); }}
table.fa-table td.fa-ok {{ color: {ACCENT}; font-weight: 700; }}
table.fa-table td.fa-no {{ color: #FF6B5E; font-weight: 700; }}
.fa-pts {{ display: flex; align-items: center; gap: 8px; }}
.fa-pts i {{ display: block; height: 7px; border-radius: 99px; background: linear-gradient(90deg, {CYAN}, {ACCENT}); }}
.fa-hbar {{ display: grid; grid-template-columns: 150px 1fr 60px; align-items: center; gap: 12px; margin: 10px 0; color: {TEXT}; font-size: .92rem; }}
.fa-hbar i {{ display: block; height: 10px; border-radius: 99px; background: linear-gradient(90deg, {CYAN}, {ACCENT}); box-shadow: 0 0 18px rgba(61,245,160,.25); }}
.fa-hbar b {{ font-family: 'Unbounded', sans-serif; font-size: .85rem; text-align: right; }}

/* steps, evidence, footer */
.fa-step {{ display: flex; gap: 18px; align-items: flex-start; }}
.fa-step .n {{ font-family: 'Unbounded', sans-serif; font-weight: 900; font-size: 1.6rem; color: transparent;
  -webkit-text-stroke: 1.2px {ACCENT}; min-width: 56px; line-height: 1; }}
.fa-step h4 {{ font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: 1rem; margin: 0 0 6px; color: {TEXT}; text-transform: uppercase; }}
.fa-step p {{ margin: 0; color: #C7CDE0; line-height: 1.6; font-size: .95rem; }}
.fa-list {{ margin: 10px 0 0; padding-left: 18px; color: #C7CDE0; line-height: 1.7; }}
.fa-foot {{ margin-top: 70px; padding: 28px 0 10px; border-top: 1px solid rgba(255,255,255,.08); display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}

/* Streamlit widgets in the same style */
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input, .stTextInput input {{
  background: rgba(255,255,255,.05) !important; border-radius: 14px !important; border-color: rgba(255,255,255,.12) !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
  background: rgba(255,255,255,.04); border-radius: 999px; padding: 8px 14px; }}
.stTabs [aria-selected="true"] {{ background: {TEXT} !important; color: #07070C !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stExpander"] {{ background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.10) !important; border-radius: 18px; }}
[data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; }}
.stDownloadButton button {{ border-radius: 999px; }}

/* animated aurora behind everything (same colours, slowly drifting) */
[data-testid="stAppViewContainer"]::after {{ content: ""; position: fixed; inset: -25%; pointer-events: none; z-index: 0;
  background: radial-gradient(28% 26% at 22% 24%, rgba(61,245,160,.16), transparent 70%),
              radial-gradient(26% 32% at 78% 30%, rgba(124,58,237,.24), transparent 70%),
              radial-gradient(34% 26% at 50% 82%, rgba(20,184,166,.17), transparent 70%),
              radial-gradient(18% 18% at 88% 78%, rgba(56,213,255,.10), transparent 70%);
  filter: blur(30px); animation: fa-aurora 28s ease-in-out infinite alternate; }}
@keyframes fa-aurora {{ 0% {{ transform: translate(0,0) rotate(0deg) scale(1); }}
  50% {{ transform: translate(3%,-2%) rotate(7deg) scale(1.07); }} 100% {{ transform: translate(-3%,3%) rotate(-5deg) scale(1.03); }} }}

/* reveal on scroll (Chrome/Edge; other browsers simply show everything) */
@supports (animation-timeline: view()) {{
  .fa-glass, .fa-card, .fa-sec, .fa-spot-wrap {{ animation: fa-reveal linear both; animation-timeline: view(); animation-range: entry 0% cover 22%; }}
}}
@keyframes fa-reveal {{ from {{ opacity: 0; translate: 0 28px; }} to {{ opacity: 1; translate: 0 0; }} }}

/* photos, flags, credits */
.fa-photo {{ width: 52px; height: 52px; border-radius: 16px; object-fit: cover; object-position: 50% 15%; flex-shrink: 0;
  border: 1px solid rgba(255,255,255,.22); background: rgba(255,255,255,.06); }}
.fa-flag {{ width: 20px; height: 14px; border-radius: 3px; object-fit: cover; vertical-align: -2px; margin-right: 7px;
  box-shadow: 0 0 0 1px rgba(255,255,255,.18); }}
.fa-credit {{ position: relative; margin-top: 10px; font-size: .6rem; line-height: 1.3; color: rgba(154,163,191,.75);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fa-credit a {{ color: rgba(154,163,191,.9) !important; }}

/* spotlight: floating photo cards */
.fa-spot-wrap {{ perspective: 900px; }}
.fa-spot {{ position: relative; border-radius: 24px; overflow: hidden; border: 1px solid rgba(255,255,255,.14); margin-bottom: 18px;
  background: linear-gradient(170deg, rgba(255,255,255,.10), rgba(255,255,255,.02)); backdrop-filter: blur(18px);
  animation: fa-float 7s ease-in-out infinite; box-shadow: 0 30px 70px rgba(0,0,0,.45); }}
.fa-spot.d1 {{ animation-delay: -2.3s; }} .fa-spot.d2 {{ animation-delay: -4.6s; }}
@keyframes fa-float {{ 0%, 100% {{ transform: translateY(0) rotate(-1.2deg); }} 50% {{ transform: translateY(-12px) rotate(1.2deg); }} }}
.fa-spot .pic {{ height: 230px; background: radial-gradient(80% 90% at 50% 100%, rgba(61,245,160,.22), transparent 70%), rgba(255,255,255,.03);
  display: flex; align-items: flex-end; justify-content: center; position: relative; }}
.fa-spot .pic img {{ height: 100%; width: 100%; object-fit: cover; object-position: 50% 18%; }}
.fa-spot .pic .ini {{ font-family: 'Unbounded', sans-serif; font-weight: 900; font-size: 4.2rem; color: rgba(255,255,255,.14); align-self: center; }}
.fa-spot .pic::after {{ content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, transparent 45%, rgba(5,5,8,.92)); }}
.fa-spot .body {{ position: relative; margin-top: -64px; padding: 0 18px 18px; z-index: 1; }}
.fa-spot .fee {{ font-family: 'Unbounded', sans-serif; font-weight: 900; font-size: 1.9rem; color: {ACCENT}; }}

/* league badges */
.fa-leagues {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 0; }}
.fa-heat {{ display: inline-block; min-width: 46px; padding: 3px 8px; border-radius: 8px; text-align: right; font-variant-numeric: tabular-nums; }}
.fa-zone-top td:first-child {{ box-shadow: inset 3px 0 0 {ACCENT}; }}
.fa-zone-down td:first-child {{ box-shadow: inset 3px 0 0 #FF6B5E; }}
.fa-range {{ color: {MUTED}; font-size: .78rem; margin-left: 8px; }}
.fa-move {{ display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.06); font-size: .9rem; }}
.fa-move:last-child {{ border-bottom: 0; }}
.fa-league {{ display: inline-flex; align-items: center; gap: 2px; padding: 8px 13px; border-radius: 999px; font-size: .8rem; font-weight: 600;
  color: {TEXT}; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); backdrop-filter: blur(10px); }}
@media (max-width: 760px) {{ .fa-nav .links {{ display: none; }} .fa-hbar {{ grid-template-columns: 100px 1fr 54px; }} }}
</style>
"""

BALL = ("<svg width='22' height='22' viewBox='0 0 34 34' aria-hidden='true'><circle cx='17' cy='17' r='15' fill='none' "
        f"stroke='{ACCENT}' stroke-width='2.8'/><path d='M17 8l7 5-3 8h-8l-3-8z' fill='{ACCENT}'/></svg>")

SECTIONS = [("forecast", "Forecast 26/27"), ("signings", "Signings 2026"), ("lab", "Transfer Lab"),
            ("evidence", "Track record"), ("method", "Method")]


def esc(x) -> str:
    return _html.escape(str(x))


def initials(name: str) -> str:
    parts = [p for p in str(name).replace("-", " ").split() if p[:1].isalpha()]
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else (parts[0][:2].upper() if parts else "?")


def nav() -> str:
    links = "".join(f"<a href='#{i}'>{esc(t)}</a>" for i, t in SECTIONS)
    return (f"<div class='fa-progress'></div><nav class='fa-nav' aria-label='Sections'>"
            f"<a class='brand' href='#top'>{BALL}FREEAGENT</a><div class='links'>{links}</div>"
            f"<a class='cta' href='#evidence'>Track record ↗</a></nav><div id='top'></div>")


def hero(meta_left: str, meta_mid: str, meta_right: str, lead: str) -> str:
    return (f"<div class='fa-meta'><span><span class='fa-dot'></span>{esc(meta_left)}</span><span class='line'></span>"
            f"<span>{esc(meta_mid)}</span><span class='line'></span><span>{esc(meta_right)}</span></div>"
            "<h1 class='fa-display'>The club<br><span class='acc'>is yours.</span></h1>"
            f"<p class='fa-lead'>{esc(lead)}</p>"
            "<a class='fa-btn primary' href='#forecast'>See the 2026/27 forecast ↓</a>"
            "<a class='fa-btn ghost' href='#lab'>Open the Transfer Lab ↓</a>")


def rows(items: list[tuple[str, str]], highlight: int | None = None) -> str:
    out = []
    for i, (k, v) in enumerate(items):
        col = f" style='color:{ACCENT}'" if i == highlight else ""
        out.append(f"<div class='r'><span class='fa-label'>{esc(k)}</span><span class='v'{col}>{esc(v)}</span></div>")
    return f"<div class='fa-glass fa-rows'>{''.join(out)}</div>"


def ticker(words: list[str]) -> str:
    seq = "".join(f"<span>{esc(w)}</span><b>◆</b>" for w in words)
    return f"<div class='fa-ticker' aria-hidden='true'><div class='t'>{seq}{seq}</div></div>"


def section(num: str, anchor: str, title: str, right: str, text: str = "") -> str:
    p = f"<p>{esc(text)}</p>" if text else ""
    return (f"<div class='fa-sec' id='{anchor}'><div class='top'><span class='n'>{esc(num)}</span><span class='line'></span>"
            f"<span>{esc(right)}</span></div><h2>{esc(title)}</h2>{p}</div>")


def way(num: str, label: str, title: str, text: str, link: str, anchor: str, colour: str) -> str:
    return (f"<a href='#{anchor}'><div class='fa-glass fa-lift fa-way' style='background:linear-gradient(160deg,{colour}1F,rgba(255,255,255,.02))'>"
            f"<div class='fa-label' style='color:{colour}'>{esc(num)} — {esc(label)}</div><h3>{esc(title)}</h3>"
            f"<p>{esc(text)}</p><div class='go' style='color:{colour}'>{esc(link)} ↗</div></div></a>")


def stat_card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    tint = " style='border-color:rgba(61,245,160,.45);background:linear-gradient(160deg,rgba(61,245,160,.14),rgba(255,255,255,.02))'" if accent else ""
    col = f" style='color:{ACCENT}'" if accent else ""
    return (f"<div class='fa-glass fa-lift'{tint}><div class='fa-label'>{esc(label)}</div>"
            f"<div class='fa-big'{col}>{esc(value)}</div><div class='fa-sub'>{esc(sub)}</div></div>")


def glass(inner_html: str) -> str:
    return f"<div class='fa-glass'>{inner_html}</div>"


def badge(verdict: str) -> str:
    c = VERDICT.get(verdict, MUTED)
    return f"<span class='fa-badge' style='color:{c};border:1px solid {c};background:{c}1F'>{esc(verdict)}</span>"


def prob_bar(p_over: float, p_fair: float, p_best: float) -> str:
    parts = []
    for name, p in (("Overpaid", p_over), ("Fair", p_fair), ("Best Buy", p_best)):
        if p <= 0.005:
            continue
        label = f"{name} {100 * p:.0f}%" if p > 0.18 else (f"{100 * p:.0f}%" if p > 0.06 else "")
        parts.append(f"<div style='flex:{p:.4f};background:{VERDICT[name]}'>{label}</div>")
    return f"<div class='fa-bar'>{''.join(parts)}</div>"


def _flag(code) -> str:
    return f"<img class='fa-flag' src='https://flagcdn.com/w40/{esc(code)}.png' alt=''>" if isinstance(code, str) and code else ""


def _face(r, cls: str = "fa-photo") -> str:
    photo = getattr(r, "photo", "") or ""
    if photo:
        return f"<img class='{cls}' src='{esc(photo)}' alt='{esc(r.player_name)}' loading='lazy'>"
    return f"<div class='fa-avatar'>{esc(initials(r.player_name))}</div>"


def _credit(r) -> str:
    c = getattr(r, "photo_credit", "") or ""
    page = getattr(r, "photo_page", "") or ""
    if not c:
        return ""
    return f"<div class='fa-credit'><a href='{esc(page)}' target='_blank' rel='noopener'>{esc(c)}</a></div>"


def player_card(r) -> str:
    """A compact signing card. r: a row of app_data.load_signings()."""
    c = VERDICT.get(r.shown_verdict, MUTED)
    pts = "n/a" if r.shown_verdict == "Not rated" else f"{r.points_mean:+.1f}"
    club = "" if str(r.to_club_name) == "nan" else r.to_club_name
    return ("<div class='fa-card'>"
            f"<div class='glow' style='background:{c}'></div>"
            f"<div class='head'>{_face(r)}"
            f"<div><div class='fa-name'>{_flag(getattr(r, 'flag', None))}{esc(r.player_name)}</div>"
            f"<div class='fa-sub'>{esc(club)} · {esc(r.season_label)}</div></div></div>"
            f"<div style='margin-top:14px;position:relative'><span class='fa-chip' style='color:{c};border-color:{c}88'>{esc(r.shown_verdict)}</span></div>"
            "<div class='fa-stats'>"
            f"<div class='fa-stat'><b>€{r.fee_m:.0f}M</b><span>Fee</span></div>"
            f"<div class='fa-stat'><b>{pts}</b><span>Points</span></div>"
            f"<div class='fa-stat'><b>{100 * r.p_regular:.0f}%</b><span>Regular</span></div>"
            f"</div>{_credit(r)}</div>")


def spotlight_card(r, delay: int = 0) -> str:
    """Large floating photo card for the hero spotlight."""
    c = VERDICT.get(r.shown_verdict, MUTED)
    photo = getattr(r, "photo", "") or ""
    pic = (f"<img src='{esc(photo)}' alt='{esc(r.player_name)}' loading='lazy'>" if photo
           else f"<span class='ini'>{esc(initials(r.player_name))}</span>")
    club = "" if str(r.to_club_name) == "nan" else r.to_club_name
    return (f"<div class='fa-spot-wrap'><div class='fa-spot d{delay}'><div class='pic'>{pic}</div><div class='body'>"
            f"<div class='fa-name' style='font-size:1.05rem'>{_flag(getattr(r, 'flag', None))}{esc(r.player_name)}</div>"
            f"<div class='fa-sub'>{esc(club)} · {esc(r.season_label)}</div>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:12px'>"
            f"<span class='fee'>€{r.fee_m:.0f}M</span><span class='fa-chip' style='color:{c};border-color:{c}88'>{esc(r.shown_verdict)}</span></div>"
            f"{_credit(r)}</div></div></div>")


def league_badges(names: dict) -> str:
    """names: league code -> display name. Each badge shows the league's country flag (no league logos)."""
    from .flags import LEAGUE_FLAG
    return "<div class='fa-leagues'>" + "".join(
        f"<span class='fa-league'>{_flag(LEAGUE_FLAG.get(k))}{esc(v)}</span>" for k, v in names.items()) + "</div>"


def season_table(t) -> str:
    """t: output of app_data.season_table(). Rendered as a broadcast-style table."""
    mx = max(float(t["Predicted points"].max()), 1.0)
    out = []
    for i, r in enumerate(t.itertuples(index=False), 1):
        ok = r[6] == "Yes"
        out.append(
            f"<tr><td style='color:{MUTED}'>{i:02d}</td><td><b>{esc(r[0])}</b></td>"
            f"<td><div class='fa-pts'><i style='width:{90 * r[1] / mx:.0f}px'></i>{r[1]:.1f}</div></td>"
            f"<td>{r[2]}%</td><td>{r[3]}%</td><td>{esc(r[4])}</td><td>{esc(r[5])}</td>"
            f"<td class='{'fa-ok' if ok else 'fa-no'}'>{'✓' if ok else '✗'}</td>"
            f"<td>{r[7]}</td><td>{r[8]}</td></tr>")
    head = ("<tr><th>#</th><th>Club</th><th>Predicted pts</th><th>Top half</th><th>Relegation</th>"
            "<th>We said</th><th>Actual</th><th></th><th>Actual pts</th><th>Final pos.</th></tr>")
    return f"<div class='fa-glass' style='overflow-x:auto'><table class='fa-table'>{head}{''.join(out)}</table></div>"


def hbars(values: dict, suffix: str = "%") -> str:
    out = [f"<div class='fa-hbar'><span>{esc(k)}</span><i style='width:{max(v, 0):.0f}%'></i><b>{v:.1f}{suffix}</b></div>"
           for k, v in values.items()]
    return glass("".join(out))


def step(n: int, title: str, text: str) -> str:
    return (f"<div class='fa-glass fa-lift'><div class='fa-step'><div class='n'>{n:02d}</div>"
            f"<div><h4>{esc(title)}</h4><p>{esc(text)}</p></div></div></div>")


def evidence(title: str, items: list[str], colour: str) -> str:
    li = "".join(f"<li>{esc(x)}</li>" for x in items)
    return (f"<div class='fa-glass' style='border-color:{colour}55;background:linear-gradient(160deg,{colour}17,rgba(255,255,255,.02))'>"
            f"<div class='fa-label' style='color:{colour}'>{esc(title)}</div><ul class='fa-list'>{li}</ul></div>")


def footer(left: str) -> str:
    return (f"<div class='fa-foot'><span>{esc(left)}</span>"
            f"<a href='#top' style='color:{ACCENT}'>Back to top ↑</a></div>")


def _heat(p: float, colour: str) -> str:
    """A percentage cell whose background strength follows the probability."""
    a = int(min(max(p, 0), 100) * 2.2)
    return f"<span class='fa-heat' style='background:{colour}{a:02X}'>{p:.0f}%</span>" if p >= 1 else \
        f"<span class='fa-heat' style='color:{MUTED}'>{'<1%' if p > 0 else '0%'}</span>"


def forecast_table(t) -> str:
    """t: output of app_data.forecast_table(). Top 4 and the bottom 3 are marked like a broadcast table."""
    mx = max(float(t["Points"].max()), 1.0)
    n = len(t)
    out = []
    for r in t.itertuples(index=False):
        zone = " class='fa-zone-top'" if r[0] <= 4 else (" class='fa-zone-down'" if r[0] > n - 3 else "")
        out.append(
            f"<tr{zone}><td style='color:{MUTED}'>{r[0]:02d}</td><td><b>{esc(r[1])}</b></td>"
            f"<td><div class='fa-pts'><i style='width:{90 * r[2] / mx:.0f}px'></i>{r[2]:.0f}</div>"
            f"<span class='fa-range'>{esc(r[3])}</span></td>"
            f"<td>{_heat(r[4], ACCENT)}</td><td>{_heat(r[5], CYAN)}</td><td>{_heat(r[6], VIOLET)}</td>"
            f"<td>{_heat(r[7], '#FF6B5E')}</td></tr>")
    head = ("<tr><th>#</th><th>Club</th><th>Points · 80% range</th><th>Title</th><th>Top 4</th><th>Top half</th>"
            "<th>Relegation</th></tr>")
    return f"<div class='fa-glass' style='overflow-x:auto'><table class='fa-table'>{head}{''.join(out)}</table></div>"


def moves(title: str, items: list[tuple[str, str, float]], colour: str) -> str:
    """items: (player, other club, fee in EUR M)."""
    body = "".join(f"<div class='fa-move'><span><b>{esc(n)}</b> <span style='color:{MUTED}'>· {esc(c)}</span></span>"
                   f"<span style='color:{colour}'>€{f:.0f}M</span></div>" for n, c, f in items) or \
        f"<div class='fa-sub'>No paid transfers in our data yet.</div>"
    return glass(f"<div class='fa-label' style='color:{colour};margin-bottom:6px'>{esc(title)}</div>{body}")
