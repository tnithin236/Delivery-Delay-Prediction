"""Delivery Delay Prediction System - Streamlit dashboard.

    streamlit run app.py

The dataset and model are created automatically on first run if missing.
Requires Streamlit >= 1.40 (keyed containers and material icons on buttons).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st

from src import visuals
from src.data_generation import (
    CATEGORIES, CUSTOMER_TYPES, DAYS, DESTINATIONS, MONTHS, SHIPPING_MODES,
    TRAFFIC_LEVELS, WAREHOUSES, WEATHER_CONDITIONS, estimate_distance,
)
from src.preprocessing import FEATURE_COLUMNS, MODEL_PATH, TARGET, load_clean_data
from src.train_model import train_and_save

st.set_page_config(page_title="Delivery Delay Prediction", page_icon="📦", layout="wide")

PAGES = ["Dashboard", "Predict Delivery Delay", "Model Performance"]
NAV_ICONS = {
    "Dashboard": ":material/home:",
    "Predict Delivery Delay": ":material/center_focus_strong:",
    "Model Performance": ":material/bar_chart:",
}
REPO_URL = "https://github.com/tnithin236/Delivery-Delay-Prediction"
DEPLOY_URL = "https://docs.streamlit.io/deploy/streamlit-community-cloud"

PRIMARY, INK, MUTED = "#2D4FE3", "#141B3A", "#6B7691"
GREEN, AMBER, RED = "#1B7F45", "#C97A00", "#C2543F"

# --------------------------------------------------------------------------- #
# Inline SVG icons
# --------------------------------------------------------------------------- #
ICON_PATHS = {
    "pin": '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
    "truck": '<rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/>'
             '<circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    "cloud": '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>'
             '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<line x1="12" y1="6" x2="12" y2="14"/><line x1="12" y1="18" x2="12.01" y2="18"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "send": '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 1.9) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[name]}</svg>')


def logo_svg(size: int = 44) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 48 48">'
            '<polygon points="24,3 43,13.5 24,24 5,13.5" fill="#8FA8FF"/>'
            '<polygon points="5,13.5 24,24 24,45 5,34.5" fill="#3F5CE0"/>'
            '<polygon points="43,13.5 24,24 24,45 43,34.5" fill="#6A4FD8"/></svg>')


PROMO_SVG = (
    '<svg viewBox="0 0 220 96" width="100%" height="96">'
    '<ellipse cx="60" cy="80" rx="34" ry="6" fill="#C9CFF5"/>'
    '<ellipse cx="165" cy="80" rx="30" ry="6" fill="#C9CFF5"/>'
    '<path d="M62 74 C 95 96, 118 44, 150 70" fill="none" stroke="#7C93F0" stroke-width="2.5" '
    'stroke-dasharray="2 6" stroke-linecap="round"/>'
    '<path d="M60 12c-14 0-24 10-24 23 0 17 24 38 24 38s24-21 24-38C84 22 74 12 60 12z" fill="#3F5CE0"/>'
    '<circle cx="60" cy="35" r="9" fill="#FFFFFF"/>'
    '<polygon points="165,34 190,46 165,58 140,46" fill="#F4C48A"/>'
    '<polygon points="140,46 165,58 165,84 140,72" fill="#E7A257"/>'
    '<polygon points="190,46 165,58 165,84 190,72" fill="#D98B3A"/></svg>'
)


def _html(*parts: str) -> str:
    """Join HTML fragments without whitespace (Markdown treats indented lines as code)."""
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
.stApp { background: #F7F8FC; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1340px; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E6EAF2; }

/* ---------- sidebar ---------- */
.brand { display: flex; align-items: center; gap: 12px; padding: .2rem .2rem 0 .2rem; }
.brand-name { font-weight: 800; color: #141B3A; font-size: 1.12rem; line-height: 1.22; }
.brand-sub { color: #6B7691; font-size: .9rem; margin: .5rem 0 1.4rem 3.7rem; }
[class*="st-key-nav"] { margin-bottom: -.35rem; }
[class*="st-key-nav"] button { width: 100%; justify-content: flex-start; gap: .55rem; border: none; box-shadow: none;
  border-radius: 10px; height: 2.9rem; padding: 0 .95rem; }
[class*="st-key-nav"] button p { font-weight: 600; font-size: .98rem; color: inherit; }
[class*="st-key-navoff_"] button { background: transparent; color: #2A3550; }
[class*="st-key-navoff_"] button:hover { background: #F1F4FB; color: #2D4FE3; }
[class*="st-key-navon_"] button { background: #E8EEFF; color: #2D4FE3; }
[class*="st-key-navon_"] button:hover { background: #E8EEFF; color: #2D4FE3; }
.model-box { margin-top: 1.5rem; padding-top: 1.2rem; border-top: 1px solid #E6EAF2; }
.mb-label { font-size: .7rem; letter-spacing: .06em; color: #8A94AB; font-weight: 600; text-transform: uppercase; }
.mb-name { display: flex; justify-content: space-between; align-items: center; margin: .6rem 0 .9rem 0;
  font-weight: 700; color: #141B3A; font-size: 1.02rem; }
.pill { font-size: .75rem; font-weight: 600; padding: .15rem .7rem; border-radius: 99px; }
.pill-active { background: #E3F5EA; color: #1B7F45; }
.mb-row { display: flex; justify-content: space-between; font-size: .86rem; color: #5B6680; padding: .28rem 0; }
.mb-row b { color: #141B3A; font-weight: 600; }
.mb-row b.good { color: #1B7F45; }
.promo { margin-top: 1.5rem; padding: 14px 16px 16px; border-radius: 14px; background: #ECEEFC; }
.promo-text { font-weight: 700; color: #141B3A; font-size: .95rem; line-height: 1.4; margin-top: .5rem; }
.promo a { display: inline-block; margin-top: .9rem; color: #2D4FE3 !important; font-weight: 600;
  font-size: .9rem; text-decoration: none; }

/* ---------- page header ---------- */
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; gap: 1rem; }
.ph-left { display: flex; align-items: center; gap: 16px; }
.ph-icon { width: 54px; height: 54px; border-radius: 14px; background: #ECEEFC; display: flex;
  align-items: center; justify-content: center; flex: none; }
.page-title { font-size: 1.95rem; font-weight: 800; color: #141B3A; line-height: 1.15; letter-spacing: -.01em; }
.page-sub { color: #6B7691; font-size: .98rem; margin-top: .3rem; }
.ph-right { display: flex; align-items: center; gap: 14px; }
.deploy { display: inline-flex; align-items: center; gap: .55rem; color: #FFFFFF !important; font-weight: 600;
  padding: .7rem 1.25rem; border-radius: 10px; text-decoration: none; font-size: .98rem;
  background: linear-gradient(90deg, #2D4FE3, #3F63F2); box-shadow: 0 4px 12px rgba(45,79,227,.25); }
.avatar { width: 40px; height: 40px; border-radius: 50%; background: #141B3A; color: #FFFFFF; display: flex;
  align-items: center; justify-content: center; font-weight: 700; }

/* ---------- input cards ---------- */
[class*="st-key-card_"] { background: #FFFFFF; border: 1px solid #E3E8F2; border-radius: 14px;
  padding: 0 22px 22px 22px; min-height: 425px; box-shadow: 0 1px 3px rgba(20,27,58,.04); gap: .55rem; }
.card-head { display: flex; align-items: center; gap: 12px; margin: 0 -22px .5rem -22px; padding: 18px 22px;
  border-bottom: 1px solid #ECEFF6; }
.num { width: 32px; height: 32px; border-radius: 50%; background: #2D4FE3; color: #fff; font-weight: 700;
  display: flex; align-items: center; justify-content: center; flex: none; font-size: .95rem; }
.card-title { font-weight: 700; color: #141B3A; font-size: 1.02rem; line-height: 1.2; }
.card-sub { color: #6B7691; font-size: .82rem; margin-top: .15rem; }
[data-testid="stWidgetLabel"] p { font-size: .86rem; font-weight: 500; color: #2A3550; }
[data-baseweb="select"] > div { background: #FFFFFF; border-color: #DCE2EC; border-radius: 8px; min-height: 42px; }
[data-baseweb="input"], [data-baseweb="base-input"] { background: #FFFFFF; border-radius: 8px; }
[data-testid="stNumberInputContainer"] { border-radius: 8px; }
[data-baseweb="input"] { border-color: #DCE2EC; }

/* ---------- predict button ---------- */
.st-key-predict button { height: 3rem; padding: 0 1.5rem; border-radius: 10px; border: none; gap: .6rem;
  background: linear-gradient(90deg, #2D4FE3, #3F63F2); box-shadow: 0 4px 12px rgba(45,79,227,.25); }
.st-key-predict button:hover { background: linear-gradient(90deg, #2543C4, #3556DC); }
.st-key-predict button p { color: #FFFFFF; font-weight: 700; font-size: 1rem; }
.st-key-predict button p::after { content: "\\2192"; margin-left: .7rem; }
.st-key-predict button span { color: #FFFFFF; }

/* ---------- result panel ---------- */
.res { display: grid; grid-template-columns: minmax(350px, 1.45fr) 1.15fr 1.4fr; border-radius: 14px; border: 1.5px solid;
  overflow: hidden; margin-top: .4rem; }
.res-ok { background: linear-gradient(100deg, #E4F4EB 0%, #F6FBF8 55%, #FFFFFF 100%); border-color: #8FCBA8; }
.res-bad { background: linear-gradient(100deg, #FBE7E2 0%, #FEF6F4 55%, #FFFFFF 100%); border-color: #E5A498; }
.res-col { padding: 26px 26px; }
.res-col + .res-col { border-left: 1px solid rgba(20,27,58,.08); }
.res-left { display: flex; align-items: center; gap: 22px; }
.seal { width: 90px; height: 90px; border-radius: 50%; flex: none; display: flex; align-items: center;
  justify-content: center; box-shadow: 0 8px 18px rgba(20,27,58,.16); }
.seal-ok { background: radial-gradient(circle at 35% 30%, #63CC8B, #1E8F4E); }
.seal-bad { background: radial-gradient(circle at 35% 30%, #E4806D, #B0432F); }
.seal-in { width: 58px; height: 58px; border-radius: 50%; background: #FFFFFF; display: flex;
  align-items: center; justify-content: center; }
.res-small { font-size: .82rem; color: #4E5A75; }
.res-status { font-size: 1.75rem; font-weight: 800; line-height: 1.15; margin: .15rem 0 .8rem 0; }
.res-ok .res-status, .res-ok .res-prob { color: #1B7F45; }
.res-bad .res-status, .res-bad .res-prob { color: #B0432F; }
.res-pline { display: flex; align-items: center; gap: .5rem; font-size: .84rem; color: #4E5A75; white-space: nowrap; }
.res-prob { font-size: 2.7rem; font-weight: 800; line-height: 1.1; margin-top: .1rem; }
.pill-low { background: #E3F5EA; color: #1B7F45; }
.pill-mid { background: #FFF0D3; color: #C97A00; }
.pill-high { background: #FFE3D6; color: #C0561F; }
.pill-vhigh { background: #FBD9D3; color: #B0432F; }
.res-mid-title { font-size: .88rem; font-weight: 600; color: #2A3550; }
.bar { position: relative; height: 10px; background: #E2E6EE; border-radius: 99px; margin: 2.6rem 0 .55rem 0; }
.bar-fill { height: 100%; border-radius: 99px; }
.bar-tip { position: absolute; bottom: 20px; transform: translateX(-50%); color: #FFFFFF; font-size: .8rem;
  font-weight: 700; padding: .2rem .65rem; border-radius: 6px; }
.bar-tip::after { content: ""; position: absolute; left: 50%; bottom: -5px; margin-left: -5px; border: 5px solid transparent;
  border-bottom: 0; border-top-color: var(--tip); }
.bar-scale { display: flex; justify-content: space-between; font-size: .78rem; color: #6B7691; }
.res-foot { font-size: .78rem; color: #4E5A75; margin-top: 1.1rem; line-height: 1.5; }
.rt-row { display: grid; grid-template-columns: 24px 150px 1fr; align-items: center; padding: .55rem 0;
  border-bottom: 1px solid rgba(20,27,58,.07); font-size: .88rem; color: #2A3550; }
.rt-row:last-child { border-bottom: none; }
.rt-row .k { font-weight: 600; color: #141B3A; }
.rt-row svg { color: #55627F; }
.empty-res { border: 1.5px dashed #C9D2E6; border-radius: 14px; padding: 26px 30px; color: #6B7691; background: #FFFFFF;
  margin-top: .4rem; font-size: .95rem; }
.info-bar { display: flex; align-items: center; gap: 14px; margin-top: 1.1rem; padding: 16px 22px; border-radius: 12px;
  background: #F1F4FD; border: 1px solid #DCE4F7; font-size: .88rem; color: #3A4666; }
.info-bar svg { color: #2D4FE3; flex: none; }

/* ---------- dashboard / performance ---------- */
.kpi { background: #FFFFFF; border: 1px solid #E3E8F2; border-radius: 14px; padding: 16px 18px; height: 100%;
  box-shadow: 0 1px 3px rgba(20,27,58,.04); }
.kpi-label { color: #6B7691; font-size: .86rem; font-weight: 500; display: flex; align-items: center; gap: .5rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.kpi-value { color: #141B3A; font-size: 1.9rem; font-weight: 800; line-height: 1.25; margin-top: .2rem; }
.kpi-note { color: #8794A6; font-size: .8rem; margin-top: .1rem; }
.section-title { font-size: 1.05rem; font-weight: 700; color: #141B3A; margin: 1.6rem 0 .7rem 0; }
[class*="st-key-chart_"] { background: #FFFFFF; border: 1px solid #E3E8F2; border-radius: 14px; padding: 12px 16px;
  box-shadow: 0 1px 3px rgba(20,27,58,.04); }

@media (max-width: 1000px) {
  .res { grid-template-columns: 1fr; }
  .res-col + .res-col { border-left: none; border-top: 1px solid rgba(20,27,58,.08); }
  .page-header { flex-direction: column; align-items: flex-start; }
}
</style>
"""

WEATHER_ICON_SVG = {
    "Clear": ("#F5A623", "<circle cx='12' cy='12' r='4'/><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4"
                         "M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/>"),
    "Cloudy": ("#7C8AA5", "<path d='M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z'/>"),
    "Rain": ("#3B72B4", "<path d='M18 10h-1.26A8 8 0 1 0 9 16h9a3 3 0 0 0 0-6z'/><path d='M8 19v2M12 19v2M16 19v2'/>"),
    "Fog": ("#7C8AA5", "<path d='M3 8h18M5 12h14M3 16h18M7 20h10'/>"),
    "Storm": ("#C2543F", "<path d='M13 2L4 14h7l-1 8 9-12h-7z'/>"),
}
TRAFFIC_ICON_SVG = ("#3B5BDB", "<line x1='12' y1='20' x2='12' y2='10'/><line x1='18' y1='20' x2='18' y2='4'/>"
                               "<line x1='6' y1='20' x2='6' y2='14'/>")


def _icon_uri(color: str, inner: str) -> str:
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='{color}' "
           f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{inner}</svg>")
    return "data:image/svg+xml," + quote(svg)


def select_icon_css(key: str, color: str, inner: str) -> str:
    """CSS that draws a small icon inside the closed selectbox with the given widget key."""
    return (f'.st-key-{key} [data-baseweb="select"] > div {{ padding-left: 32px; background-repeat: no-repeat; '
            f'background-position: 11px center; background-size: 18px; '
            f'background-image: url("{_icon_uri(color, inner)}"); }}')


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Preparing the model for first use. This takes under a minute...")
def load_bundle() -> dict:
    """Load the saved model; train it first if it is missing or was built with another scikit-learn."""
    bundle = None
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("sklearn_version") != sklearn.__version__:
                bundle = None
        except Exception:
            bundle = None
    if bundle is None:
        bundle = train_and_save(verbose=False)
    return bundle


@st.cache_data(show_spinner="Loading delivery data...")
def get_data() -> pd.DataFrame:
    return load_clean_data()


# --------------------------------------------------------------------------- #
# UI building blocks
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str) -> None:
    st.markdown(_html(
        '<div class="page-header"><div class="ph-left"><div class="ph-icon">', logo_svg(30), '</div><div>',
        f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div></div></div>',
        '<div class="ph-right">',
        f'<a class="deploy" href="{DEPLOY_URL}" target="_blank" rel="noopener">', icon("send", 18, "#FFFFFF"), 'Deploy</a>',
        '<div class="avatar">A</div></div></div>'), unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", accent: str = PRIMARY) -> None:
    st.markdown(_html(
        f'<div class="kpi"><div class="kpi-label"><span class="dot" style="background:{accent}"></span>{label}</div>',
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'), unsafe_allow_html=True)


def show_chart(fig, key: str) -> None:
    """Render a Matplotlib figure inside a white card, then free it."""
    with st.container(key=f"chart_{key}"):
        st.pyplot(fig)
    plt.close(fig)


def card_head(num: int, icon_name: str, title: str, subtitle: str) -> None:
    st.markdown(_html(
        f'<div class="card-head"><span class="num">{num}</span>', icon(icon_name, 22, PRIMARY),
        f'<div><div class="card-title">{title}</div><div class="card-sub">{subtitle}</div></div></div>'),
        unsafe_allow_html=True)


def go_to(page: str) -> None:
    st.session_state["page"] = page


def render_sidebar(bundle: dict) -> None:
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%d %b %Y") if MODEL_PATH.exists() else "-"
    with st.sidebar:
        st.markdown(_html(
            '<div class="brand">', logo_svg(46), '<div class="brand-name">Delivery Delay<br>Prediction</div></div>',
            '<div class="brand-sub">Logistics analytics</div>'), unsafe_allow_html=True)

        for i, name in enumerate(PAGES):
            active = st.session_state["page"] == name
            st.button(name, key=f"{'navon' if active else 'navoff'}_{i}", icon=NAV_ICONS[name],
                      on_click=go_to, args=(name,))

        st.markdown(_html(
            '<div class="model-box"><div class="mb-label">Active model</div>',
            f'<div class="mb-name"><span>{bundle["model_name"]}</span><span class="pill pill-active">Active</span></div>',
            f'<div class="mb-row"><span>F1 Score</span><b class="good">{bundle["metrics"]["F1 Score"]:.1%}</b></div>',
            f'<div class="mb-row"><span>Trained on</span><b>{bundle["n_train"]:,} orders</b></div>',
            f'<div class="mb-row"><span>Last trained</span><b>{trained}</b></div></div>',
            '<div class="promo">', PROMO_SVG,
            '<div class="promo-text">Smarter predictions<br>for faster deliveries.</div>',
            f'<a href="{REPO_URL}" target="_blank" rel="noopener">Learn more &nbsp;&rarr;</a></div>'),
            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard(df: pd.DataFrame, bundle: dict) -> None:
    page_header("Delivery Performance Dashboard",
                "How orders across the network are performing, and where delays concentrate.")

    total = len(df)
    delayed = int(df[TARGET].sum())
    on_time = total - delayed
    cols = st.columns(5)
    with cols[0]:
        kpi("Total Orders", f"{total:,}", "Unique orders after cleaning", INK)
    with cols[1]:
        kpi("On-Time Deliveries", f"{on_time:,}", f"{on_time / total:.1%} of orders", PRIMARY)
    with cols[2]:
        kpi("Delayed Deliveries", f"{delayed:,}", f"{delayed / total:.1%} of orders", RED)
    with cols[3]:
        kpi("Delay Rate", f"{delayed / total:.1%}", "Delayed / total orders", MUTED)
    with cols[4]:
        kpi("Model Accuracy", f"{bundle['metrics']['Accuracy']:.1%}",
            f"{bundle['model_name']}, held-out test set", PRIMARY)

    section_title("Delivery performance")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(visuals.plot_delivery_split(df), "split")
    with c2:
        show_chart(visuals.plot_delay_by_mode(df), "mode")
    c3, c4 = st.columns(2)
    with c3:
        show_chart(visuals.plot_delay_by_traffic(df), "traffic")
    with c4:
        show_chart(visuals.plot_delay_by_weather(df), "weather")
    c5, c6 = st.columns(2)
    with c5:
        show_chart(visuals.plot_distance_vs_delay(df), "distance")
    with c6:
        show_chart(visuals.plot_delay_by_month(df), "month")


def sync_distance() -> None:
    """Refill the distance field with a typical road distance when the route changes."""
    st.session_state["distance"] = estimate_distance(
        st.session_state["origin"], st.session_state["destination"])


def risk_badge(p: float) -> tuple[str, str]:
    if p < 0.25:
        return "Low risk", "pill-low"
    if p < 0.50:
        return "Moderate risk", "pill-mid"
    if p < 0.75:
        return "High risk", "pill-high"
    return "Very high risk", "pill-vhigh"


def result_panel(prob: float, delayed: bool, ctx: dict) -> None:
    """Three-panel result: status, probability breakdown, and input summary."""
    tone = "res-bad" if delayed else "res-ok"
    color = "#B0432F" if delayed else GREEN
    status = "DELIVERY DELAYED" if delayed else "ON TIME"
    badge_text, badge_cls = risk_badge(prob)
    pct = round(prob * 100)
    tip_pos = min(max(pct, 5), 95)
    seal_icon = icon("alert", 34, "#B0432F", 2.6) if delayed else icon("check", 34, GREEN, 3)

    rows = [
        ("pin", "Route", f'{ctx["origin"]} &rarr; {ctx["destination"]} &nbsp;({ctx["distance"]:,} km)'),
        ("truck", "Shipping", f'{ctx["mode"]} &bull; {ctx["customer"]} customer'),
        ("cloud", "Conditions", f'{ctx["weather"]} weather, {ctx["traffic"]} traffic'),
        ("clock", "Processing Time", f'{ctx["processing"]:.2f} hrs'),
        ("globe", "Model", ctx["model"]),
    ]
    table = "".join(f'<div class="rt-row">{icon(i, 18)}<span class="k">{k}</span><span>{v}</span></div>'
                    for i, k, v in rows)

    st.markdown(_html(
        f'<div class="res {tone}">',
        # left: status
        '<div class="res-col res-left">',
        f'<div class="seal {"seal-bad" if delayed else "seal-ok"}"><div class="seal-in">{seal_icon}</div></div>',
        '<div><div class="res-small">Predicted Delivery Status</div>',
        f'<div class="res-status">{status}</div>',
        f'<div class="res-pline">Delay Probability <span class="pill {badge_cls}">{badge_text}</span></div>',
        f'<div class="res-prob">{pct}%</div></div></div>',
        # middle: breakdown
        '<div class="res-col"><div class="res-mid-title">Probability Breakdown</div>',
        f'<div class="bar"><div class="bar-fill" style="width:{pct}%;background:{color}"></div>',
        f'<div class="bar-tip" style="left:{tip_pos}%;--tip:{color};background:{color}">{pct}%</div></div>',
        '<div class="bar-scale"><span>0%</span><span>50%</span><span>100%</span></div>',
        '<div class="res-foot">Based on historical delivery patterns and current inputs, there is a ',
        f'<b style="color:{color}">{pct}%</b> chance of delay.</div></div>',
        # right: summary
        f'<div class="res-col">{table}</div></div>',
        '<div class="info-bar">', icon("info", 20),
        f'<span>Estimated by the {ctx["model"]} model from historical delivery patterns. '
        'Treat it as a risk score, not a guarantee.</span></div>'), unsafe_allow_html=True)


def page_predict(bundle: dict) -> None:
    page_header("Predict Delivery Delay",
                "Enter the order and shipping details to estimate whether it will arrive on time.")

    if "distance" not in st.session_state:
        st.session_state["distance"] = estimate_distance(WAREHOUSES[0], "Chennai")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1, st.container(key="card_route"):
        card_head(1, "pin", "Route &amp; Order", "Where the parcel starts, ends and details.")
        a, b = st.columns(2)
        with a:
            origin = st.selectbox("Warehouse / Origin", WAREHOUSES, key="origin", on_change=sync_distance)
        with b:
            destination = st.selectbox("Destination", DESTINATIONS, index=DESTINATIONS.index("Chennai"),
                                       key="destination", on_change=sync_distance)
        a, b = st.columns(2)
        with a:
            distance = st.number_input("Distance (km)", min_value=5, max_value=3500, step=5, key="distance")
        with b:
            category = st.selectbox("Product Category", CATEGORIES)
        a, b = st.columns(2)
        with a:
            order_value = st.number_input("Order Value (₹)", min_value=50.0, max_value=200000.0,
                                          value=2500.0, step=100.0)
        with b:
            n_items = st.number_input("Number of Items", min_value=1, max_value=20, value=2, step=1)

    with col2, st.container(key="card_shipping"):
        card_head(2, "truck", "Shipping &amp; Customer", "How the order is shipped and who placed it.")
        mode = st.selectbox("Shipping Mode", SHIPPING_MODES, index=1)
        customer = st.selectbox("Customer Type", CUSTOMER_TYPES, index=1)
        a, b = st.columns(2)
        with a:
            cost = st.number_input("Shipping Cost (₹)", min_value=0.0, max_value=5000.0, value=180.0, step=10.0)
        a, b = st.columns(2)
        with a:
            day = st.selectbox("Order Day", DAYS, index=2)
        with b:
            month = st.selectbox("Order Month", MONTHS, index=5)

    with col3, st.container(key="card_conditions"):
        card_head(3, "cloud", "Conditions &amp; History", "Operating conditions along the route.")
        weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=0, key="weather")
        traffic = st.selectbox("Traffic Level", TRAFFIC_LEVELS, index=1, key="traffic")
        a, b = st.columns(2)
        with a:
            prev_delays = st.number_input("Previous Delays", min_value=0, max_value=10, value=0, step=1,
                                          help="Delayed orders this customer had in the last 12 months.")
        a, b = st.columns(2)
        with a:
            processing = st.number_input("Processing Time (hrs)", min_value=0.5, max_value=120.0, value=18.0,
                                         step=0.5, help="Hours from order placement until the parcel leaves the warehouse.")

    # small icons inside the weather and traffic dropdowns
    w_color, w_inner = WEATHER_ICON_SVG[weather]
    st.markdown("<style>" + select_icon_css("weather", w_color, w_inner)
                + select_icon_css("traffic", *TRAFFIC_ICON_SVG) + "</style>", unsafe_allow_html=True)

    st.write("")
    clicked = st.button("Predict Delivery Status", key="predict", icon=":material/online_prediction:")

    if clicked:
        row = {
            "Warehouse_Origin": origin, "Destination": destination, "Product_Category": category,
            "Order_Value": float(order_value), "Distance_km": float(distance),
            "Shipping_Mode": mode, "Customer_Type": customer, "Order_Day": day, "Order_Month": month,
            "Weather_Condition": weather, "Traffic_Level": traffic, "Number_of_Items": int(n_items),
            "Previous_Delays": int(prev_delays), "Processing_Time_hrs": float(processing),
            "Shipping_Cost": float(cost),
        }
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        pipeline = bundle["pipeline"]
        prob = float(pipeline.predict_proba(X)[0, 1])
        delayed = bool(pipeline.predict(X)[0] == 1)
        result_panel(prob, delayed, {
            "origin": origin, "destination": destination, "distance": int(distance), "mode": mode,
            "customer": customer, "weather": weather, "traffic": traffic, "processing": float(processing),
            "model": bundle["model_name"],
        })
    else:
        st.markdown('<div class="empty-res">Fill in the order details above, then select '
                    '<b>Predict Delivery Status</b> to see the estimated delivery outcome and delay probability.</div>',
                    unsafe_allow_html=True)


def page_performance(bundle: dict) -> None:
    page_header("Model Performance",
                f"{bundle['model_name']} had the best F1 score on {bundle['n_test']:,} held-out test orders.")

    m = bundle["metrics"]
    cols = st.columns(4)
    with cols[0]:
        kpi("Accuracy", f"{m['Accuracy']:.1%}", "Share of all predictions that were correct", INK)
    with cols[1]:
        kpi("Precision", f"{m['Precision']:.1%}", "Predicted delays that were real", PRIMARY)
    with cols[2]:
        kpi("Recall", f"{m['Recall']:.1%}", "Real delays the model caught", PRIMARY)
    with cols[3]:
        kpi("F1 Score", f"{m['F1 Score']:.1%}", "Balance of precision and recall", MUTED)

    section_title("Where the model is right and wrong")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(visuals.plot_confusion_matrix(bundle["confusion_matrix"]), "cm")
    with c2:
        show_chart(visuals.plot_feature_importance(pd.DataFrame(bundle["feature_importance"])), "fi")

    section_title("Model comparison")
    comparison = pd.DataFrame(bundle["comparison"])
    table = comparison.copy()
    table["Selected"] = table["Model"].map(lambda name: "Yes" if name == bundle["model_name"] else "")
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    c3, c4 = st.columns(2)
    with c3:
        with st.container(key="chart_table"):
            st.dataframe(table, hide_index=True)
            st.caption("Both models use class weighting so delays, the minority class, are not overlooked. "
                       "The best model is chosen by F1 score, which balances precision and recall.")
    with c4:
        show_chart(visuals.plot_model_comparison(comparison), "compare")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.session_state.setdefault("page", PAGES[0])
    st.markdown(CSS, unsafe_allow_html=True)
    bundle = load_bundle()
    df = get_data()
    render_sidebar(bundle)

    page = st.session_state["page"]
    if page == PAGES[0]:
        page_dashboard(df, bundle)
    elif page == PAGES[1]:
        page_predict(bundle)
    else:
        page_performance(bundle)


main()
