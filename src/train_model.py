"""Train, compare and save the delivery delay model.

    python src/train_model.py

Steps: load data -> describe -> clean -> EDA figures -> encode + split ->
train Logistic Regression and Random Forest -> compare -> pick best by F1 ->
compute feature importance -> save everything with Joblib.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import visuals  # noqa: E402
from src.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS, FIGURES_DIR, MODEL_PATH, TARGET, build_preprocessor, clean_data,
    describe_dataset, load_data,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_models() -> dict:
    """The candidate models. Class weights compensate for delays being the minority class."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Standard classification metrics on the held-out test set."""
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, proba),
    }


def save_eda_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Write the EDA charts to ``reports/figures``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "01_delivery_split.png": visuals.plot_delivery_split,
        "02_delay_by_shipping_mode.png": visuals.plot_delay_by_mode,
        "03_delay_by_traffic.png": visuals.plot_delay_by_traffic,
        "04_delay_by_weather.png": visuals.plot_delay_by_weather,
        "05_distance_vs_delay.png": visuals.plot_distance_vs_delay,
        "06_delay_by_month.png": visuals.plot_delay_by_month,
    }
    paths = []
    for name, fn in charts.items():
        fig = fn(df)
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_dir / name)
    return paths


def _save_model_figures(bundle: dict, out_dir: Path = FIGURES_DIR) -> None:
    figs = {
        "07_feature_importance.png": visuals.plot_feature_importance(pd.DataFrame(bundle["feature_importance"])),
        "08_confusion_matrix.png": visuals.plot_confusion_matrix(bundle["confusion_matrix"]),
        "09_model_comparison.png": visuals.plot_model_comparison(pd.DataFrame(bundle["comparison"])),
    }
    for name, fig in figs.items():
        fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def train_and_save(model_path: Path = MODEL_PATH, verbose: bool = True, save_figures: bool = True) -> dict:
    """Run the full pipeline and save the model bundle. Returns the bundle."""
    log = print if verbose else (lambda *a, **k: None)

    # 1-2. Load and describe
    log("=" * 64, "\n1. LOAD DATASET\n" + "=" * 64)
    raw = load_data()
    if verbose:
        describe_dataset(raw)

    # 3-4. Missing values and duplicates
    log("\n" + "=" * 64, "\n2. CLEAN DATA\n" + "=" * 64)
    df = clean_data(raw)
    log(f"Removed {len(raw) - len(df):,} duplicate rows -> {len(df):,} unique orders.")
    log(f"Missing feature values ({int(df[FEATURE_COLUMNS].isna().sum().sum()):,} cells) are imputed inside the "
        "model pipeline (median for numeric, most frequent for categorical).")

    # 5. EDA
    if save_figures:
        log("\n" + "=" * 64, "\n3. EXPLORATORY DATA ANALYSIS\n" + "=" * 64)
        paths = save_eda_figures(df)
        log(f"Saved {len(paths)} charts to {FIGURES_DIR}")

    # 6-7. Split (encoding happens inside the pipeline)
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log("\n" + "=" * 64, "\n4. TRAIN / TEST SPLIT\n" + "=" * 64)
    log(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows   (stratified, {TEST_SIZE:.0%} test)")

    # 8-9. Train and compare
    log("\n" + "=" * 64, "\n5. TRAIN AND COMPARE MODELS\n" + "=" * 64)
    fitted, results = {}, []
    for name, model in get_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append({"Model": name, **evaluate(pipe, X_test, y_test)})
    comparison = pd.DataFrame(results)
    log(comparison.round(4).to_string(index=False))

    # 10. Select by F1
    best_name = comparison.loc[comparison["F1 Score"].idxmax(), "Model"]
    best_pipe = fitted[best_name]
    best_metrics = comparison.set_index("Model").loc[best_name].to_dict()
    log(f"\nBest model by F1 score: {best_name} (F1 = {best_metrics['F1 Score']:.4f})")

    cm = confusion_matrix(y_test, best_pipe.predict(X_test), labels=[0, 1])
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="f1", n_repeats=8,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": perm.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))
    log("\nTop features:")
    log(importance.head(6).round(4).to_string(index=False))

    # 11. Save
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "comparison": comparison.to_dict("records"),
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance.to_dict("records"),
        "feature_columns": FEATURE_COLUMNS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    log(f"\nModel saved to {model_path}")

    if save_figures:
        _save_model_figures(bundle)
        log(f"Model charts saved to {FIGURES_DIR}")
    return bundle


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # no GUI windows when run from the terminal
    train_and_save()
