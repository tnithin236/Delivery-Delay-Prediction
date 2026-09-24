# Delivery Delay Prediction System

An end-to-end machine learning application that predicts whether an e-commerce delivery will arrive **On Time** or be **Delayed**, with a Streamlit analytics dashboard for exploring delivery performance and scoring new orders.

## Project Overview

The project covers the whole ML workflow in a small, readable codebase: data generation, cleaning, exploratory analysis, model training and comparison, model persistence, and an interactive web app. It runs locally with no external downloads and no manual setup.

## Problem Statement

Late deliveries hurt customer satisfaction and increase support costs. Logistics teams usually learn about a delay only after it happens. If the risk can be estimated when an order is placed, the team can switch shipping modes, prioritise processing or warn the customer in advance.

## Objectives

- Predict `Delivery_Delayed` (0 = On Time, 1 = Delayed) from order and shipping information.
- Compare Logistic Regression and Random Forest and select the better model by F1 score.
- Show which factors drive delays.
- Provide a clean dashboard where a user can enter an order and get a delay probability.

## Features

- **Dashboard**: total orders, on-time and delayed counts, delay rate, model accuracy, and six charts of delivery performance.
- **Predict Delivery Delay**: a form with all 15 input features and a card-style result showing **ON TIME** or **DELIVERY DELAYED** plus the delay probability.
- **Model Performance**: accuracy, precision, recall, F1, confusion matrix, feature importance and a model comparison.
- **Self-setting-up**: if the dataset or trained model is missing, the app creates them on first run.

## Dataset Description

No public dataset ships with the project, so `src/data_generation.py` creates a realistic **synthetic dataset of 6,000 unique orders** (plus about 1% duplicate rows and about 1% missing values so the cleaning step is meaningful). It simulates an Indian logistics network with 6 warehouses and 18 destination cities. Delay probability depends on distance, traffic, weather, shipping mode, season, processing time, customer history and a few interactions (for example storms on long routes), plus random noise. The overall delay rate is about 30%.

| Column | Description |
|---|---|
| `Warehouse_Origin`, `Destination` | Origin warehouse and destination city |
| `Product_Category` | Electronics, Clothing, Home & Kitchen, Groceries, Books, Beauty, Furniture, Sports |
| `Order_Value` | Order value in ₹ |
| `Distance_km` | Approximate road distance |
| `Shipping_Mode` | Economy, Standard, Express, Priority |
| `Customer_Type` | New, Regular, Premium |
| `Order_Day`, `Order_Month` | Day of week and month of the order |
| `Weather_Condition` | Clear, Cloudy, Rain, Fog, Storm |
| `Traffic_Level` | Low, Medium, High |
| `Number_of_Items` | Items in the order |
| `Previous_Delays` | Delayed orders this customer had in the last 12 months |
| `Processing_Time_hrs` | Hours from order placement until the parcel leaves the warehouse |
| `Shipping_Cost` | Shipping cost in ₹ |
| `Delivery_Delayed` | **Target**: 0 = On Time, 1 = Delayed |

Because the data is synthetic, the reported metrics show how the pipeline behaves, not how it would perform on real courier data.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.

## ML Workflow

1. Load the dataset (generated automatically if missing).
2. Display dataset information: shape, dtypes, missing values, duplicates, target balance.
3. Remove duplicate rows.
4. Handle missing values inside the model pipeline (median for numeric, most frequent for categorical), so imputation is learned from the training split only.
5. Explore the data with six charts saved to `reports/figures/`.
6. Encode categorical features (one-hot) and scale numeric features.
7. Split into 80% training and 20% test data, stratified by the target.
8. Train Logistic Regression and Random Forest.
9. Compare accuracy, precision, recall and F1 on the test set.
10. Select the model with the highest F1 score.
11. Compute permutation feature importance and save the pipeline, metrics and charts data with Joblib to `models/delivery_delay_model.pkl`.

## Models Used

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Random Forest | 300 trees, `min_samples_leaf=3`, `class_weight="balanced_subsample"` |

Class weighting is used because delays are the minority class (about 30%). It raises recall, so more real delays are caught, at the cost of some precision. It also means the displayed "delay probability" works as a risk score rather than an exact real-world frequency.

## Evaluation Metrics

Results on the held-out test set (1,200 orders):

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 72.4% | 52.4% | 72.0% | **60.6%** | 79.5% |
| Random Forest | 76.6% | 61.9% | 53.7% | 57.5% | 79.2% |

Random Forest is more accurate overall but misses more real delays. Logistic Regression has the higher F1, so it is selected. Numbers may differ slightly if you retrain with different package versions.

The strongest predictors are distance, traffic level, shipping mode, product category and weather.

## Project Structure

```
delivery-delay-prediction/
├── data/
│   └── delivery_data.csv            # synthetic dataset (auto-generated if missing)
├── models/
│   └── delivery_delay_model.pkl     # trained pipeline + metrics (auto-trained if missing)
├── notebooks/
│   └── eda_and_model.ipynb          # step-by-step EDA and modelling
├── reports/
│   └── figures/                     # EDA and model charts (PNG), written by train_model.py
├── src/
│   ├── data_generation.py           # synthetic data generator
│   ├── preprocessing.py             # paths, columns, cleaning, encoding pipeline
│   ├── train_model.py               # train, compare, select, save
│   └── visuals.py                   # charts shared by the notebook, script and app
├── .streamlit/
│   └── config.toml                  # light theme
├── app.py                           # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation Steps

Python 3.9 or newer is recommended.

```bash
cd delivery-delay-prediction

# optional but recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## How to Run

Launch the app:

```bash
streamlit run app.py
```

It opens at http://localhost:8501. The dataset and model are already included; if you delete them, the app recreates them on the first run (the first launch then takes a short while).

Optional commands:

```bash
python src/data_generation.py    # regenerate data/delivery_data.csv
python src/train_model.py        # retrain, print the comparison and refresh the charts
jupyter notebook notebooks/eda_and_model.ipynb
```

## Example Prediction

Input:

| Field | Value |
|---|---|
| Route | Mumbai to Chennai, 1,290 km |
| Product / value / items | Electronics, ₹15,000, 3 items |
| Shipping mode / customer | Economy, New customer |
| Order day / month | Saturday, November |
| Weather / traffic | Storm, High |
| Previous delays | 3 |
| Processing time / shipping cost | 40 hrs, ₹150 |

Output:

```
DELIVERY DELAYED
Delay Probability: 99%
```

Changing the same order to Priority shipping, clear weather and low traffic lowers the delay probability from 99% to about 57%; the long route, new customer and slow processing still keep it elevated. A short, well-run shipment (Bengaluru to Coimbatore, Priority, clear weather, low traffic, no previous delays) scores about 1%.

## Future Improvements

- Train on real courier or marketplace data and validate with time-based splits.
- Tune hyperparameters and try gradient boosting (XGBoost, LightGBM).
- Calibrate probabilities and let users choose the decision threshold.
- Add SHAP explanations to show why a specific order is flagged.
- Fetch live weather and traffic data instead of manual inputs.
- Add batch scoring by CSV upload and a REST API for order systems.
- Add automated tests and a scheduled retraining job.
