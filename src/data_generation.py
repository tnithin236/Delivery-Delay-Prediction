"""Synthetic e-commerce delivery data for the Delivery Delay Prediction System.

The generator simulates an Indian e-commerce logistics network (6 warehouses,
18 destination cities). Delay probability is driven by realistic factors
(distance, traffic, weather, shipping mode, seasonality, processing time,
customer history ...) plus unobserved noise, so the prediction task is
learnable but not trivial.

Run directly to (re)create ``data/delivery_data.csv``:

    python src/data_generation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "delivery_data.csv"

# --------------------------------------------------------------------------- #
# Reference data (also used by the Streamlit app for its input widgets)
# --------------------------------------------------------------------------- #
CITY_COORDS = {
    "Ahmedabad": (23.023, 72.572),
    "Bengaluru": (12.972, 77.594),
    "Bhopal": (23.260, 77.413),
    "Chandigarh": (30.733, 76.779),
    "Chennai": (13.083, 80.271),
    "Coimbatore": (11.017, 76.956),
    "Delhi": (28.614, 77.209),
    "Hyderabad": (17.385, 78.487),
    "Indore": (22.720, 75.858),
    "Jaipur": (26.912, 75.787),
    "Kochi": (9.932, 76.267),
    "Kolkata": (22.573, 88.364),
    "Lucknow": (26.847, 80.947),
    "Mumbai": (19.076, 72.878),
    "Nagpur": (21.146, 79.088),
    "Patna": (25.594, 85.138),
    "Pune": (18.520, 73.857),
    "Visakhapatnam": (17.687, 83.219),
}
WAREHOUSES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"]
DESTINATIONS = sorted(CITY_COORDS)
METRO_CITIES = {"Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"}

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Groceries",
    "Books", "Beauty", "Furniture", "Sports",
]
SHIPPING_MODES = ["Economy", "Standard", "Express", "Priority"]
CUSTOMER_TYPES = ["New", "Regular", "Premium"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]

ROAD_FACTOR = 1.25  # straight-line distance -> approximate road distance


def estimate_distance(origin: str, destination: str) -> int:
    """Typical road distance in km between two cities (used as a form default)."""
    if origin == destination:
        return 25
    lat1, lon1 = map(np.radians, CITY_COORDS[origin])
    lat2, lon2 = map(np.radians, CITY_COORDS[destination])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return int(round(2 * 6371 * np.arcsin(np.sqrt(a)) * ROAD_FACTOR))


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisection search for the intercept that yields the target delay rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_dataset(
    n_orders: int = 6000,
    seed: int = 42,
    delay_rate: float = 0.30,
    missing_rate: float = 0.012,
    duplicate_rate: float = 0.01,
) -> pd.DataFrame:
    """Create a realistic synthetic delivery dataset.

    ``n_orders`` unique orders are generated. A small share of values is then
    blanked out and a small share of rows duplicated, so the cleaning steps in
    the ML pipeline have real work to do.
    """
    rng = np.random.default_rng(seed)
    n = n_orders

    # --- Route ------------------------------------------------------------- #
    origin = rng.choice(WAREHOUSES, n, p=[0.22, 0.22, 0.18, 0.14, 0.12, 0.12])
    dest_w = np.array([2.0 if d in METRO_CITIES else 1.0 for d in DESTINATIONS])
    destination = rng.choice(DESTINATIONS, n, p=dest_w / dest_w.sum())

    base_km = np.array([estimate_distance(o, d) for o, d in zip(origin, destination)], dtype=float)
    distance = np.where(
        origin == destination,
        rng.uniform(8, 45, n),
        base_km * rng.normal(1.0, 0.04, n),
    ).round(1)

    # --- Customer, product, shipping mode ---------------------------------- #
    customer = rng.choice(CUSTOMER_TYPES, n, p=[0.25, 0.55, 0.20])
    mode_probs = {
        "New": [0.30, 0.55, 0.13, 0.02],
        "Regular": [0.25, 0.50, 0.20, 0.05],
        "Premium": [0.05, 0.35, 0.35, 0.25],
    }
    mode = np.empty(n, dtype=object)
    for ctype, probs in mode_probs.items():
        mask = customer == ctype
        mode[mask] = rng.choice(SHIPPING_MODES, mask.sum(), p=probs)

    category = rng.choice(CATEGORIES, n, p=[0.20, 0.20, 0.15, 0.12, 0.08, 0.09, 0.06, 0.10])
    cat_items = {"Electronics": 1.0, "Clothing": 1.8, "Home & Kitchen": 1.6, "Groceries": 5.0,
                 "Books": 1.5, "Beauty": 1.4, "Furniture": 0.2, "Sports": 1.3}
    cat_value = {"Electronics": 12000, "Clothing": 1800, "Home & Kitchen": 3000, "Groceries": 900,
                 "Books": 600, "Beauty": 1200, "Furniture": 14000, "Sports": 2500}
    items = np.clip(1 + rng.poisson([cat_items[c] for c in category]), 1, 20)
    order_value = (
        np.array([cat_value[c] for c in category])
        * rng.lognormal(0, 0.55, n) * (0.7 + 0.3 * items)
    ).round(0)

    # --- Calendar ---------------------------------------------------------- #
    day = rng.choice(DAYS, n, p=[0.14, 0.13, 0.13, 0.14, 0.15, 0.16, 0.15])
    month_p = np.array([0.075, 0.07, 0.075, 0.075, 0.075, 0.075,
                        0.075, 0.075, 0.08, 0.11, 0.12, 0.10])
    month = rng.choice(MONTHS, n, p=month_p / month_p.sum())
    festive = np.isin(month, ["October", "November", "December"])

    # --- Weather (seasonal) ------------------------------------------------ #
    weather = np.empty(n, dtype=object)
    season_probs = {
        "monsoon": (["June", "July", "August", "September"], [0.22, 0.26, 0.42, 0.00, 0.10]),
        "winter": (["December", "January", "February"], [0.35, 0.22, 0.05, 0.33, 0.05]),
        "other": (None, [0.50, 0.28, 0.12, 0.05, 0.05]),
    }
    assigned = np.zeros(n, dtype=bool)
    for months_in_season, probs in season_probs.values():
        mask = ~assigned if months_in_season is None else np.isin(month, months_in_season)
        weather[mask] = rng.choice(WEATHER_CONDITIONS, mask.sum(), p=probs)
        assigned |= mask

    # --- Traffic: latent congestion score cut into 35/40/25 % bands -------- #
    is_weekday = np.isin(day, DAYS[:5])
    is_metro_dest = np.isin(destination, list(METRO_CITIES))
    congestion = rng.normal(0, 1, n) + 0.5 * is_metro_dest + 0.3 * is_weekday
    low_cut, high_cut = np.quantile(congestion, [0.35, 0.75])
    traffic = np.where(congestion < low_cut, "Low", np.where(congestion < high_cut, "Medium", "High"))

    # --- Operations -------------------------------------------------------- #
    wh_proc = {"Mumbai": 1.0, "Delhi": 2.0, "Bengaluru": -1.0, "Chennai": 0.0,
               "Kolkata": 3.0, "Hyderabad": -0.5}
    cat_proc = {"Electronics": 4, "Clothing": 0, "Home & Kitchen": 2, "Groceries": -2,
                "Books": -1, "Beauty": 0, "Furniture": 12, "Sports": 1}
    processing = (
        (rng.gamma(3.0, 4.5, n) + [cat_proc[c] for c in category] + [wh_proc[o] for o in origin])
        * np.where(festive, 1.25, 1.0)
    )
    processing = np.clip(processing, 1, 96).round(1)

    prev_lambda = np.select([customer == "New", customer == "Regular"], [0.5, 0.9], default=0.7)
    previous_delays = np.clip(rng.poisson(prev_lambda), 0, 8)

    mode_cost = {"Economy": 0.8, "Standard": 1.0, "Express": 1.6, "Priority": 2.3}
    shipping_cost = (
        (35 + 0.09 * distance + 12 * items + 0.0015 * order_value)
        * np.array([mode_cost[m] for m in mode])
        * np.where(customer == "Premium", 0.9, 1.0)
        * rng.lognormal(0, 0.10, n)
    ).round(0)

    # --- Delay model (logistic, with interactions and hidden noise) -------- #
    effects = {
        "mode": {"Economy": 0.45, "Standard": 0.0, "Express": -0.45, "Priority": -0.75},
        "traffic": {"Low": -0.5, "Medium": 0.0, "High": 0.85},
        "weather": {"Clear": -0.25, "Cloudy": 0.0, "Rain": 0.55, "Fog": 0.6, "Storm": 1.2},
        "category": {"Furniture": 0.35, "Electronics": 0.1, "Groceries": -0.1},
        "warehouse": {"Kolkata": 0.25, "Delhi": 0.15, "Mumbai": 0.10, "Bengaluru": -0.15,
                      "Chennai": -0.10, "Hyderabad": -0.10},
        "customer": {"Premium": -0.2, "New": 0.15},
        "month": {"June": 0.15, "July": 0.35, "August": 0.35, "September": 0.30,
                  "October": 0.35, "November": 0.50, "December": 0.45},
    }
    z = (
        1.3 * distance / 1000
        + np.array([effects["mode"][m] for m in mode])
        + np.array([effects["traffic"][t] for t in traffic])
        + np.array([effects["weather"][w] for w in weather])
        + np.array([effects["category"].get(c, 0.0) for c in category])
        + np.array([effects["warehouse"][o] for o in origin])
        + np.array([effects["customer"].get(c, 0.0) for c in customer])
        + np.array([effects["month"].get(m, 0.0) for m in month])
        + 0.035 * (processing - 18)
        + 0.28 * previous_delays
        + 0.05 * (items - 2)
        + 0.08 * np.log(order_value / 2500)
        + 0.20 * np.isin(day, ["Saturday", "Sunday"])
        # interactions
        + 0.50 * ((weather == "Storm") & (distance > 1000))
        + 0.30 * ((traffic == "High") & (mode == "Economy"))
        + 0.25 * ((traffic == "High") & is_metro_dest)
        + rng.normal(0, 0.25, n)  # unobserved factors
    )
    z = z + _calibrate_intercept(z, delay_rate)
    delayed = (rng.random(n) < _sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "Order_ID": [f"ORD-{100001 + i}" for i in range(n)],
        "Warehouse_Origin": origin,
        "Destination": destination,
        "Product_Category": category,
        "Order_Value": order_value,
        "Distance_km": distance,
        "Shipping_Mode": mode,
        "Customer_Type": customer,
        "Order_Day": day,
        "Order_Month": month,
        "Weather_Condition": weather,
        "Traffic_Level": traffic,
        "Number_of_Items": items,
        "Previous_Delays": previous_delays,
        "Processing_Time_hrs": processing,
        "Shipping_Cost": shipping_cost,
        "Delivery_Delayed": delayed,
    })

    # --- Real-world messiness: missing values and duplicated rows ---------- #
    for col in ["Order_Value", "Weather_Condition", "Traffic_Level",
                "Processing_Time_hrs", "Shipping_Cost"]:
        df.loc[rng.random(n) < missing_rate, col] = np.nan

    duplicates = df.sample(int(n * duplicate_rate), random_state=seed)
    df = pd.concat([df, duplicates]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def create_dataset_file(path: Path = DATA_PATH, **kwargs) -> pd.DataFrame:
    """Generate the dataset and write it to ``path``."""
    df = generate_dataset(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def ensure_dataset(path: Path = DATA_PATH) -> Path:
    """Return the dataset path, generating the CSV first if it doesn't exist."""
    if not path.exists():
        print(f"[data] {path.name} not found - generating synthetic dataset ...")
        df = create_dataset_file(path)
        print(f"[data] Saved {len(df):,} rows to {path}")
    return path


if __name__ == "__main__":
    out = create_dataset_file()
    print(f"Created {DATA_PATH} with {len(out):,} rows x {out.shape[1]} columns")
    print(f"Delay rate: {out['Delivery_Delayed'].mean():.1%}")
    sys.exit(0)
