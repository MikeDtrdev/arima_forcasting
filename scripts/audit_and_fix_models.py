import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "tondo_crime_data_barangay_41_43_2019_2025.csv"
MODELS_DIR = ROOT / "tondo_forecasts" / "models"


def list_models():
    models = []
    if not MODELS_DIR.exists():
        return models
    for fn in os.listdir(MODELS_DIR):
        if fn.startswith("model_") and fn.endswith(".pkl"):
            parts = fn[6:-4].split("__")
            if len(parts) == 2:
                crime_type = parts[0].replace("_", " ")
                location = parts[1].replace("_", " ")
                models.append((crime_type, location, MODELS_DIR / fn))
    return models


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


def forecast_from_model(model, steps=12):
    try:
        fc = model.get_forecast(steps=steps).predicted_mean
    except Exception:
        try:
            fc = model.forecast(steps=steps)
        except Exception:
            return None
    fc = np.clip(np.array(fc, dtype=float), 0, None)
    return fc


def is_degenerate(values: np.ndarray) -> bool:
    if values is None or len(values) == 0:
        return True
    if np.allclose(values, 0.0):
        return True
    # Constant or near-constant series
    if float(np.ptp(values)) < 1e-6:
        return True
    return False


def grid_search_sarimax(y: pd.Series):
    best_aic = np.inf
    best_fit = None
    best_cfg = None
    p_vals = [0, 1]
    d_vals = [0, 1]
    q_vals = [0, 1]
    P_vals = [0, 1]
    D_vals = [1]
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
                                    enforce_invertibility=True,
                                )
                                res = model.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_fit = res
                                    best_cfg = ((p, d, q), (P, D, Q, m))
                            except Exception:
                                continue
    return best_fit, best_cfg


def retrain_model(crime_type: str, location: str, out_path: Path):
    y = load_monthly_counts(crime_type, location)
    fit, cfg = grid_search_sarimax(y)
    if fit is None:
        # Fallback
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            trend='c',
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fit = model.fit(disp=False)
        cfg = ((1, 1, 1), (0, 1, 1, 12))
    steps = 12
    fc = np.clip(np.array(fit.get_forecast(steps=steps).predicted_mean, dtype=float), 0, None)
    with open(out_path, 'wb') as f:
        pickle.dump(fit, f)
    return cfg, float(fit.aic), fc.tolist()


def main():
    models = list_models()
    if not models:
        print("No models found.")
        return
    print(f"Found {len(models)} models. Auditing...")
    retrained = []
    for crime_type, location, path in models:
        print(f"\nModel: {crime_type} @ {location}\nPath: {path}")
        # Load existing model
        model = None
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Failed to load model: {e}")
        # Check forecast degeneracy
        deg = True
        if model is not None:
            fc = forecast_from_model(model, steps=12)
            deg = is_degenerate(fc)
            print("Existing forecast sample:", [] if fc is None else np.round(fc, 2).tolist())
        else:
            print("No existing model; will train new one.")
        if deg:
            print("Detected degenerate or missing model. Retraining...")
            cfg, aic, fc_new = retrain_model(crime_type, location, path)
            retrained.append({
                "crime_type": crime_type,
                "location": location,
                "path": str(path),
                "cfg": cfg,
                "aic": aic,
                "forecast_sample": np.round(np.array(fc_new), 2).tolist(),
            })
            print(f"Retrained with cfg={cfg}, AIC={aic:.2f}")
            print("New forecast sample:", np.round(np.array(fc_new), 2).tolist())
        else:
            print("Model looks OK; skipping retrain.")
    print(f"\nRetrained {len(retrained)} models.")
    if retrained:
        print("Summary of retrained models:")
        for r in retrained:
            print(f"- {r['crime_type']} @ {r['location']} | AIC={r['aic']:.2f} | cfg={r['cfg']} | path={r['path']}")


if __name__ == "__main__":
    main()