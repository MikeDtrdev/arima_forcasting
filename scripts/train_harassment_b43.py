import warnings
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

CRIME_TYPE = "Harassment"
LOCATION = "Barangay 43"

# Resolve paths robustly relative to repo root
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "tondo_crime_data_barangay_41_43_2019_2025.csv"
MODELS_DIR = ROOT / "tondo_forecasts" / "models"
OUT_FILE = MODELS_DIR / f"model_{CRIME_TYPE.replace(' ', '_')}__{LOCATION.replace(' ', '_')}.pkl"

warnings.filterwarnings("ignore")


def load_monthly_counts():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    if 'crimeType' in df.columns:
        df['crimeType'] = df['crimeType'].astype(str).str.strip()
    if 'location' in df.columns:
        df['location'] = df['location'].astype(str).str.strip()
    # Filter for target series
    df = df[(df['crimeType'] == CRIME_TYPE) & (df['location'] == LOCATION)].copy()
    if df.empty:
        raise ValueError(f"No rows for {CRIME_TYPE} at {LOCATION}")
    # Parse month and aggregate counts
    df['month'] = pd.to_datetime(df['dateTime'], errors='coerce').dt.to_period('M').astype(str)
    grouped = df.groupby('month').size().reset_index(name='count').sort_values('month')
    labels = grouped['month'].tolist()
    values = grouped['count'].astype(int).tolist()
    return labels, values


def fit_seasonal_arima(y):
    y = np.asarray(y, dtype=float)
    if y.size < 6:
        raise ValueError("Insufficient data points to fit ARIMA (need >= 6)")
    # Candidate grids (compact but effective)
    p = d = q = [0, 1]
    seasonal_p = seasonal_d = seasonal_q = [0, 1]
    seasonal_m = [12]
    best_aic = np.inf
    best_model = None
    best_cfg = None
    for order in itertools.product(p, d, q):
        for seas in itertools.product(seasonal_p, seasonal_d, seasonal_q, seasonal_m):
            try:
                model = SARIMAX(y, order=order, seasonal_order=seas, trend='c', enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_model = res
                    best_cfg = (order, seas)
            except Exception:
                continue
    if best_model is None:
        # Fallback simple model
        model = SARIMAX(y, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c', enforce_stationarity=False, enforce_invertibility=False)
        best_model = model.fit(disp=False)
        best_cfg = ((0, 1, 1), (1, 1, 1, 12))
    print(f"Selected order={best_cfg[0]} seasonal_order={best_cfg[1]} AIC={best_model.aic:.2f}")
    return best_model


def main():
    labels, values = load_monthly_counts()
    print(f"Loaded {len(values)} monthly counts for {CRIME_TYPE} at {LOCATION}")
    print("Last 6 months:", list(zip(labels[-6:], values[-6:])))
    model = fit_seasonal_arima(values)
    # Preview forecast
    fc = model.forecast(steps=12)
    print("Sample 12-month forecast:", [round(float(v), 4) for v in fc])
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {OUT_FILE}")


if __name__ == "__main__":
    main()