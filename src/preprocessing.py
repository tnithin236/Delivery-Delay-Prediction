"""Shared configuration, data loading/cleaning and the preprocessing pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Make ``from src...`` imports work no matter how a script is launched.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generation import DATA_PATH, ensure_dataset  # noqa: E402

MODEL_PATH = ROOT / "models" / "delivery_delay_model.pkl"
FIGURES_DIR = ROOT / "reports" / "figures"

TARGET = "Delivery_Delayed"
ID_COLUMN = "Order_ID"

CATEGORICAL_FEATURES = [
    "Warehouse_Origin", "Destination", "Product_Category", "Shipping_Mode",
    "Customer_Type", "Order_Day", "Order_Month", "Weather_Condition", "Traffic_Level",
]
NUMERIC_FEATURES = [
    "Order_Value", "Distance_km", "Number_of_Items", "Previous_Delays",
    "Processing_Time_hrs", "Shipping_Cost",
]
FEATURE_COLUMNS = [
    "Warehouse_Origin", "Destination", "Product_Category", "Order_Value", "Distance_km",
    "Shipping_Mode", "Customer_Type", "Order_Day", "Order_Month", "Weather_Condition",
    "Traffic_Level", "Number_of_Items", "Previous_Delays", "Processing_Time_hrs",
    "Shipping_Cost",
]

FEATURE_LABELS = {
    "Warehouse_Origin": "Warehouse / Origin",
    "Destination": "Destination",
    "Product_Category": "Product Category",
    "Order_Value": "Order Value (₹)",
    "Distance_km": "Distance (km)",
    "Shipping_Mode": "Shipping Mode",
    "Customer_Type": "Customer Type",
    "Order_Day": "Order Day",
    "Order_Month": "Order Month",
    "Weather_Condition": "Weather Condition",
    "Traffic_Level": "Traffic Level",
    "Number_of_Items": "Number of Items",
    "Previous_Delays": "Previous Delays",
    "Processing_Time_hrs": "Processing Time (hrs)",
    "Shipping_Cost": "Shipping Cost (₹)",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the delivery dataset, generating it first if the CSV is missing."""
    return pd.read_csv(ensure_dataset(path))


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a compact overview: shape, dtypes, missing values, duplicates, target."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    df.info()
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    print(missing.to_string() if len(missing) else "  none")
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")
    if TARGET in df:
        counts = df[TARGET].value_counts().sort_index()
        print("\nTarget balance (0 = On Time, 1 = Delayed):")
        for cls, cnt in counts.items():
            print(f"  {cls}: {cnt:,} ({cnt / len(df):.1%})")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate orders and rows without a target label.

    Missing *feature* values are deliberately kept here and imputed inside the
    model pipeline (see ``build_preprocessor``), so imputation statistics are
    learned from the training split only and the deployed model can also handle
    incomplete inputs.
    """
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset in one call."""
    return clean_data(load_data(path))


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric features, impute + one-hot encode categorical ones."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
