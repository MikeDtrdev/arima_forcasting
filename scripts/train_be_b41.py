import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

CRIME_TYPE = "Breaking and Entering"
LOCATION = "Barangay 41"

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "tondo_crime_data_barangay_41_43_2019_2025.csv"
MODEL_DIR = ROOT / "tondo_forecasts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model_Breaking_and_Entering__Barangay_41.pkl"


def load_monthly_counts(crime_type: str, location: str) -> pd.Series:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df['crimeType'] = df['crimeType'].astype(str).str.strip()
    df['location']  = df['location'].astype(str).str.strip()
    df = df[(df['crimeType'] == crime_type.strip()) & (df['location'] == location.strip())]
    if df.empty:
        raise RuntimeError(f"No records found for {crime_type} @ {location}")
    df['dt'] = pd.to_datetime(df['dateTime'], errors='coerce')
    df = df.dropna(subset=['dt'])
    df['month'] = df['dt'].dt.to_period('M').astype(str)
    grouped = df.groupby('month').size().reset_index(name='count').sort_values('month')
    start = pd.to_datetime(grouped['month'].iloc[0])
    end   = pd.to_datetime(grouped['month'].iloc[-1])
    all_months = pd.period_range(start=start, end=end, freq='M').astype(str)
    series = pd.Series(0.0, index=all_months)
    for _, row in grouped.iterrows():
        series[row['month']] = float(row['count'])
    dt_index = pd.to_datetime(series.index)
    y = pd.Series(series.values, index=dt_index)
    y.index = pd.DatetimeIndex(y.index).to_period('M').to_timestamp()
    return y


def grid_search_sarimax(y: pd.Series):
    best_aic = np.inf
    best_fit = None
    best_cfg = None
    # Focus on seasonal structure with a constant trend
    p_vals = [0, 1]
    d_vals = [0, 1]
    q_vals = [0, 1]
    P_vals = [0, 1]
    D_vals = [1]  # force seasonal differencing to capture yearly pattern
    Q_vals = [0, 1]
    m = 12
    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                for P in P_vals:
                    for D in D_vals:
                        for Q in Q_vals:
                            try:
                                model = SARIMAX(
                                    y,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, m),
                                    trend='c',
                                    enforce_stationarity=True,
                                    enforce_invertibility=True
                                )
                                res = model.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_fit = res
                                    best_cfg = ((p, d, q), (P, D, Q, m))
                            except Exception:
                                continue
    return best_fit, best_cfg


def train_and_save():
    print("Loading monthly counts...")
    y = load_monthly_counts(CRIME_TYPE, LOCATION)
    print(f"Series length: {len(y)} months; mean={y.mean():.2f}")

    print("Fitting seasonal SARIMAX via grid search...")
    fit, cfg = grid_search_sarimax(y)

    if fit is None:
        print("Grid search failed; trying fallback (1,1,1)(0,1,1)[12] trend=c")
        try:
            model = SARIMAX(
                y,
                order=(1,1,1),
                seasonal_order=(0,1,1,12),
                trend='c',
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            fit = model.fit(disp=False)
            cfg = ((1,1,1),(0,1,1,12))
        except Exception as e:
            raise RuntimeError(f"Fallback fit failed: {e}")

    print(f"Selected cfg: order={cfg[0]} seasonal_order={cfg[1]} AIC={fit.aic:.2f}")

    # Quick sanity forecast on raw scale
    steps = 12
    fc = fit.get_forecast(steps=steps).predicted_mean
    fc = np.clip(np.array(fc, dtype=float), 0, None)
    print("Sample forecast (12 months):", np.round(fc, 2).tolist())

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(fit, f)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()