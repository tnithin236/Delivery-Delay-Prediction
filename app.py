"""Delivery Delay Prediction System - Streamlit dashboard.

    streamlit run app.py

The dataset and model are created automatically on first run if missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
NAVY, BLUE, SLATE, BRICK, GREEN = "#1B2F4B", "#2F6DB5", "#64748B", "#C2543F", "#2E7D5B"

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid #DDE3EC; }

.brand { padding: .4rem 0 1.1rem 0; border-bottom: 1px solid #D5DCE6; margin-bottom: 1rem; }
.brand-name { font-size: 1.2rem; font-weight: 700; color: #1B2F4B; line-height: 1.25; }
.brand-sub { font-size: .85rem; color: #64748B; margin-top: .2rem; }
.side-note { font-size: .82rem; color: #64748B; line-height: 1.5; margin-top: 1.2rem;
  padding-top: 1rem; border-top: 1px solid #D5DCE6; }
.side-note b { color: #1B2F4B; }

.page-header { margin: 0 0 1.5rem 0; }
.page-title { font-size: 1.8rem; font-weight: 700; color: #1B2F4B; letter-spacing: -0.01em; line-height: 1.2; }
.page-sub { color: #64748B; font-size: 1rem; margin-top: .35rem; }

.kpi { background: #FFFFFF; border: 1px solid #E0E6EF; border-left: 4px solid var(--accent, #2F6DB5);
  border-radius: 10px; padding: 16px 18px; height: 100%; }
.kpi-label { color: #64748B; font-size: .86rem; font-weight: 500; }
.kpi-value { color: #1B2F4B; font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin-top: .1rem; }
.kpi-note { color: #8794A6; font-size: .8rem; margin-top: .1rem; }

.section-title { font-size: 1.05rem; font-weight: 700; color: #1B2F4B; margin: 1.6rem 0 .7rem 0; }
.group-title { font-size: .95rem; font-weight: 700; color: #1B2F4B; margin-bottom: .2rem; }
.group-sub { font-size: .82rem; color: #64748B; margin-bottom: .6rem; }

.result { border-radius: 10px; padding: 26px 30px; margin-top: 1.2rem; border: 1px solid; border-left-width: 6px; }
.result-ontime { background: #EDF6F1; border-color: #2E7D5B; }
.result-delayed { background: #FBEEEA; border-color: #C2543F; }
.result-caption { font-size: .9rem; color: #64748B; }
.result-status { font-size: 2.2rem; font-weight: 800; letter-spacing: .01em; line-height: 1.2; margin: .15rem 0 .5rem 0; }
.result-ontime .result-status { color: #1E5A41; }
.result-delayed .result-status { color: #8F3524; }
.result-prob { font-size: 1.25rem; font-weight: 600; color: #1B2F4B; }
.result-route { font-size: .9rem; color: #475467; margin-top: .9rem; }
.prob-track { background: #FFFFFF; border: 1px solid #D5DCE6; border-radius: 99px; height: 12px;
  overflow: hidden; margin-top: .7rem; max-width: 520px; }
.prob-fill { height: 100%; border-radius: 99px; }
.result-note { font-size: .82rem; color: #64748B; margin-top: .8rem; }

div.stButton > button { width: 100%; height: 3.2rem; font-size: 1.05rem; font-weight: 700;
  border-radius: 10px; background: #1F4E8C; color: #FFFFFF; border: 1px solid #1F4E8C; }
div.stButton > button:hover { background: #173B6C; border-color: #173B6C; color: #FFFFFF; }
div.stButton > button p { color: #FFFFFF; font-weight: 700; font-size: 1.05rem; }
</style>
"""


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
# UI helpers
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header"><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", accent: str = BLUE) -> None:
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_chart(fig) -> None:
    """Render a Matplotlib figure inside a bordered card, then free it."""
    with st.container(border=True):
        st.pyplot(fig)
    plt.close(fig)


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
        kpi("Total Orders", f"{total:,}", "Unique orders after cleaning", NAVY)
    with cols[1]:
        kpi("On-Time Deliveries", f"{on_time:,}", f"{on_time / total:.1%} of orders", BLUE)
    with cols[2]:
        kpi("Delayed Deliveries", f"{delayed:,}", f"{delayed / total:.1%} of orders", BRICK)
    with cols[3]:
        kpi("Delay Rate", f"{delayed / total:.1%}", "Delayed / total orders", SLATE)
    with cols[4]:
        kpi("Model Accuracy", f"{bundle['metrics']['Accuracy']:.1%}",
            f"{bundle['model_name']}, held-out test set", BLUE)

    section_title("Delivery performance")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(visuals.plot_delivery_split(df))
    with c2:
        show_chart(visuals.plot_delay_by_mode(df))

    c3, c4 = st.columns(2)
    with c3:
        show_chart(visuals.plot_delay_by_traffic(df))
    with c4:
        show_chart(visuals.plot_delay_by_weather(df))

    c5, c6 = st.columns(2)
    with c5:
        show_chart(visuals.plot_distance_vs_delay(df))
    with c6:
        show_chart(visuals.plot_delay_by_month(df))


def sync_distance() -> None:
    """Refill the distance field with a typical road distance when the route changes."""
    st.session_state["distance"] = estimate_distance(
        st.session_state["origin"], st.session_state["destination"])


def risk_label(p: float) -> str:
    if p < 0.25:
        return "Low delay risk"
    if p < 0.50:
        return "Moderate delay risk"
    if p < 0.75:
        return "High delay risk"
    return "Very high delay risk"


def result_card(prob: float, delayed: bool, summary: str, model_name: str) -> None:
    css_class = "result-delayed" if delayed else "result-ontime"
    status = "DELIVERY DELAYED" if delayed else "ON TIME"
    color = BRICK if delayed else GREEN
    st.markdown(
        f'<div class="result {css_class}">'
        f'<div class="result-caption">Predicted delivery status</div>'
        f'<div class="result-status">{status}</div>'
        f'<div class="result-prob">Delay Probability: {prob * 100:.0f}%'
        f'&nbsp;&nbsp;<span style="color:{SLATE};font-weight:500;font-size:1rem">{risk_label(prob)}</span></div>'
        f'<div class="prob-track"><div class="prob-fill" style="width:{prob * 100:.0f}%;background:{color}"></div></div>'
        f'<div class="result-route">{summary}</div>'
        f'<div class="result-note">Estimated by the {model_name} model from historical delivery patterns. '
        f'Treat it as a risk score, not a guarantee.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def page_predict(bundle: dict) -> None:
    page_header("Predict Delivery Delay",
                "Enter the order and shipping details to estimate whether it will arrive on time.")

    if "distance" not in st.session_state:
        st.session_state["distance"] = estimate_distance(WAREHOUSES[0], "Chennai")

    left, middle, right = st.columns(3, gap="medium")
    with left, st.container(border=True):
        st.markdown('<div class="group-title">Route and order</div>'
                    '<div class="group-sub">Where the parcel starts, ends and what is in it.</div>',
                    unsafe_allow_html=True)
        origin = st.selectbox("Warehouse / Origin", WAREHOUSES, key="origin", on_change=sync_distance)
        destination = st.selectbox("Destination", DESTINATIONS, index=DESTINATIONS.index("Chennai"),
                                   key="destination", on_change=sync_distance)
        distance = st.number_input("Distance (km)", min_value=5, max_value=3500, step=5, key="distance")
        category = st.selectbox("Product Category", CATEGORIES)
        order_value = st.number_input("Order Value (₹)", min_value=50.0, max_value=200000.0,
                                      value=2500.0, step=100.0)
        n_items = st.number_input("Number of Items", min_value=1, max_value=20, value=2, step=1)

    with middle, st.container(border=True):
        st.markdown('<div class="group-title">Shipping and customer</div>'
                    '<div class="group-sub">How the order is shipped and who placed it.</div>',
                    unsafe_allow_html=True)
        mode = st.selectbox("Shipping Mode", SHIPPING_MODES, index=1)
        customer = st.selectbox("Customer Type", CUSTOMER_TYPES, index=1)
        cost = st.number_input("Shipping Cost (₹)", min_value=0.0, max_value=5000.0, value=180.0, step=10.0)
        day = st.selectbox("Order Day", DAYS, index=2)
        month = st.selectbox("Order Month", MONTHS, index=5)

    with right, st.container(border=True):
        st.markdown('<div class="group-title">Conditions and history</div>'
                    '<div class="group-sub">Operating conditions along the route.</div>',
                    unsafe_allow_html=True)
        weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=0)
        traffic = st.selectbox("Traffic Level", TRAFFIC_LEVELS, index=1)
        prev_delays = st.number_input("Previous Delays", min_value=0, max_value=10, value=0, step=1,
                                      help="Delayed orders this customer had in the last 12 months.")
        processing = st.number_input("Processing Time (hrs)", min_value=0.5, max_value=120.0,
                                     value=18.0, step=0.5,
                                     help="Time from order placement until the parcel leaves the warehouse.")

    st.write("")
    if st.button("Predict Delivery Status"):
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
        summary = (f"{origin} to {destination}, {int(distance):,} km, {mode} shipping, "
                   f"{weather.lower()} weather, {traffic.lower()} traffic.")
        result_card(prob, delayed, summary, bundle["model_name"])


def page_performance(bundle: dict) -> None:
    page_header("Model Performance",
                f"{bundle['model_name']} was selected because it had the best F1 score on the held-out test set "
                f"({bundle['n_test']:,} orders, never seen during training).")

    m = bundle["metrics"]
    cols = st.columns(4)
    with cols[0]:
        kpi("Accuracy", f"{m['Accuracy']:.1%}", "Share of all predictions that were correct", NAVY)
    with cols[1]:
        kpi("Precision", f"{m['Precision']:.1%}", "Predicted delays that were real", BLUE)
    with cols[2]:
        kpi("Recall", f"{m['Recall']:.1%}", "Real delays the model caught", BLUE)
    with cols[3]:
        kpi("F1 Score", f"{m['F1 Score']:.1%}", "Balance of precision and recall", SLATE)

    section_title("Where the model is right and wrong")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(visuals.plot_confusion_matrix(bundle["confusion_matrix"]))
    with c2:
        show_chart(visuals.plot_feature_importance(pd.DataFrame(bundle["feature_importance"])))

    section_title("Model comparison")
    comparison = pd.DataFrame(bundle["comparison"])
    table = comparison.copy()
    table["Selected"] = table["Model"].map(lambda name: "Yes" if name == bundle["model_name"] else "")
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    c3, c4 = st.columns([1, 1])
    with c3:
        with st.container(border=True):
            st.dataframe(table, hide_index=True)
            st.caption("Both models use class weighting so delays, the minority class, are not overlooked. "
                       "The best model is chosen by F1 score, which balances precision and recall.")
    with c4:
        show_chart(visuals.plot_model_comparison(comparison))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    bundle = load_bundle()
    df = get_data()

    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-name">Delivery Delay Prediction</div>'
                    '<div class="brand-sub">Logistics analytics</div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.markdown(
            f'<div class="side-note">Active model<br><b>{bundle["model_name"]}</b><br>'
            f'F1 score {bundle["metrics"]["F1 Score"]:.1%}<br>'
            f'Trained on {bundle["n_train"]:,} orders</div>',
            unsafe_allow_html=True,
        )

    if page == PAGES[0]:
        page_dashboard(df, bundle)
    elif page == PAGES[1]:
        page_predict(bundle)
    else:
        page_performance(bundle)


main()
