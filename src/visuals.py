"""Chart helpers shared by the training script, the notebook and the Streamlit app.

Every function returns a Matplotlib ``Figure`` in a consistent blue/gray style.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.data_generation import MONTHS, SHIPPING_MODES, TRAFFIC_LEVELS, WEATHER_CONDITIONS
from src.preprocessing import FEATURE_LABELS, TARGET

NAVY = "#1B2F4B"
BLUE = "#3B72B4"
LIGHT_BLUE = "#A9C4E4"
SLATE = "#64748B"
GRID = "#E9EDF3"
BRICK = "#C2543F"
TEXT = "#1F2A3C"

DISTANCE_BINS = [0, 100, 300, 600, 1000, 1500, 2000, 10_000]
DISTANCE_LABELS = ["< 100", "100-300", "300-600", "600-1,000", "1,000-1,500", "1,500-2,000", "2,000+"]


def apply_style() -> None:
    """Set the global Matplotlib look (light, minimal, blue/gray)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D5DCE6",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": SLATE,
        "axes.labelsize": 9.5,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXT,
        "font.family": "sans-serif",
    })


apply_style()


def _finish(fig):
    fig.tight_layout()
    return fig


def plot_delivery_split(df: pd.DataFrame, figsize=(6, 3.6)):
    """Donut chart: on-time vs delayed deliveries."""
    counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=figsize)
    wedges, _ = ax.pie(
        counts.values, colors=[BLUE, BRICK], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0.07, f"{total:,}", ha="center", va="center", fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0, -0.15, "orders", ha="center", va="center", fontsize=9.5, color=SLATE)
    ax.legend(
        wedges,
        [f"On time    {counts[0]:,}  ({counts[0] / total:.1%})",
         f"Delayed    {counts[1]:,}  ({counts[1] / total:.1%})"],
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10,
    )
    ax.set_title("On-time vs delayed deliveries", loc="center")
    ax.grid(False)
    return _finish(fig)


def plot_delay_rate(df: pd.DataFrame, column: str, order=None, title=None,
                    xlabel=None, figsize=(6, 3.6)):
    """Bar chart of the delay rate (%) per category, with the overall rate as a reference line."""
    rate = df.groupby(column)[TARGET].mean() * 100
    if order is not None:
        rate = rate.reindex([o for o in order if o in rate.index])
    overall = df[TARGET].mean() * 100

    fig, ax = plt.subplots(figsize=figsize)
    colors = [NAVY if v == rate.max() else BLUE for v in rate.values]
    bars = ax.bar(rate.index.astype(str), rate.values, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=TEXT)
    ax.axhline(overall, color=SLATE, linestyle="--", linewidth=1, label=f"Overall rate ({overall:.0f}%)")
    ax.set_ylim(0, max(rate.max(), overall) * 1.28)
    ax.set_ylabel("Delay rate (%)")
    ax.set_xlabel(xlabel or "")
    ax.set_title(title or f"Delay rate by {column.replace('_', ' ').lower()}")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    return _finish(fig)


def plot_delay_by_mode(df, figsize=(6, 3.6)):
    return plot_delay_rate(df, "Shipping_Mode", SHIPPING_MODES, "Delivery delay by shipping mode", figsize=figsize)


def plot_delay_by_traffic(df, figsize=(6, 3.6)):
    return plot_delay_rate(df, "Traffic_Level", TRAFFIC_LEVELS, "Delivery delay by traffic level", figsize=figsize)


def plot_delay_by_weather(df, figsize=(6, 3.6)):
    return plot_delay_rate(df, "Weather_Condition", WEATHER_CONDITIONS, "Delivery delay by weather condition", figsize=figsize)


def plot_distance_vs_delay(df: pd.DataFrame, figsize=(6, 3.6)):
    """Delay rate by distance band."""
    band = pd.cut(df["Distance_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS)
    tmp = pd.DataFrame({"band": band, TARGET: df[TARGET]})
    tmp["band"] = tmp["band"].astype(str)
    fig = plot_delay_rate(tmp, "band", DISTANCE_LABELS, "Distance vs delivery delay",
                          xlabel="Distance band (km)", figsize=figsize)
    fig.axes[0].tick_params(axis="x", labelsize=8)
    return fig


def plot_delay_by_month(df: pd.DataFrame, figsize=(6, 3.6)):
    """Line chart of the delay rate through the year."""
    rate = (df.groupby("Order_Month")[TARGET].mean() * 100).reindex(MONTHS)
    fig, ax = plt.subplots(figsize=figsize)
    x = [m[:3] for m in MONTHS]
    ax.plot(x, rate.values, color=BLUE, linewidth=2.2, marker="o", markersize=5, markerfacecolor="white",
            markeredgewidth=1.8)
    ax.fill_between(x, rate.values, rate.min() * 0.6, color=LIGHT_BLUE, alpha=0.25)
    ax.axhline(df[TARGET].mean() * 100, color=SLATE, linestyle="--", linewidth=1)
    ax.set_ylim(rate.min() * 0.6, rate.max() * 1.15)
    ax.set_ylabel("Delay rate (%)")
    ax.set_title("Seasonal delay pattern")
    ax.grid(axis="x", visible=False)
    return _finish(fig)


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15, figsize=(6, 4.2)):
    """Horizontal bars of permutation importance (columns: Feature, Importance)."""
    top = importance.sort_values("Importance", ascending=False).head(top_n).iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in top["Feature"]]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, top["Importance"].clip(lower=0), color=BLUE, height=0.65)
    bars[-1].set_color(NAVY)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color=TEXT)
    ax.set_xlabel("Drop in F1 score when the feature is shuffled")
    ax.set_title("Feature importance")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, top["Importance"].clip(lower=0).max() * 1.18)
    return _finish(fig)


def plot_confusion_matrix(cm, figsize=(6, 4.2)):
    """Confusion-matrix heatmap: counts plus the share of each actual class."""
    cm = np.asarray(cm)
    share = cm / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f"{cm[i, j]:,}\n{share[i, j]:.0%} of actual" for j in range(2)] for i in range(2)])
    cmap = LinearSegmentedColormap.from_list("blues_soft", ["#F4F7FB", "#2F5C94"])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(share, annot=labels, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, square=True,
                linewidths=3, linecolor="white", annot_kws={"fontsize": 11, "fontweight": "bold"},
                xticklabels=["On Time", "Delayed"], yticklabels=["On Time", "Delayed"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    ax.grid(False)
    return _finish(fig)


def plot_model_comparison(comparison: pd.DataFrame, figsize=(6, 3.8)):
    """Grouped bars comparing models on accuracy, precision, recall and F1."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = comparison["Model"].tolist()
    colors = [BLUE, "#9AA7B8", LIGHT_BLUE]
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=figsize)
    for i, (_, row) in enumerate(comparison.iterrows()):
        vals = [row[m] * 100 for m in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92, label=row["Model"], color=colors[i % 3])
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(models), fontsize=9)
    return _finish(fig)
