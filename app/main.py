from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
import math

APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
DATA_PATH = APP_ROOT / "data" / "realistic_crimes_2019_2024.csv"

app = FastAPI(title="E-Responde Crime Forecasting (ARIMA)")


def _load_registry():
    import json

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("registry.json not found. Train models first.")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _predict_generic(model, steps: int):
    # Supports pmdarima and statsmodels SARIMAXResults
    try:
        return model.predict(n_periods=steps)  # pmdarima
    except Exception:
        try:
            return model.forecast(steps=steps)  # statsmodels results
        except Exception:
            res = model.get_forecast(steps=steps)
            return getattr(res, "predicted_mean", res)


def _sanitize_for_json(value):
    # Recursively replace NaN/Inf with None so JSON serialization won't fail
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/series")
def list_series():
    try:
        reg = _load_registry()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model registry not found. Run training to create models/registry.json.")
    items = [
        {
            "key": k,
            "barangay": v["barangay"],
            "crime_type": v["crime_type"],
            "n_obs": v["n_obs"],
            "metrics": _sanitize_for_json(v.get("metrics", {})),
        }
        for k, v in reg["series"].items()
    ]
    return {"count": len(items), "items": items}


@app.get("/forecast")
def forecast(
    barangay: str = Query(..., description="Barangay name, e.g., 'Barangay 41'"),
    crime_type: str = Query(..., description="Crime type, e.g., 'Theft'"),
    months: Optional[int] = Query(None, ge=1, le=24, description="Forecast horizon in months"),
):
    try:
        reg = _load_registry()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model registry not found. Run training to create models/registry.json.")
    key = f"{barangay}__{crime_type}"
    info = reg["series"].get(key)
    if not info:
        raise HTTPException(status_code=404, detail="Series not found. Use /series to inspect available keys.")

    model_path = Path(info["model_path"]) if Path(info["model_path"]).is_absolute() else (MODELS_DIR / Path(info["model_path"]).name)
    if not model_path.exists():
        raise HTTPException(status_code=500, detail="Model file missing. Retrain the models.")

    model = joblib.load(model_path)
    horizon = months or  reg.get("forecast_months_default", 6)
    yhat = _predict_generic(model, steps=horizon)

    # Build monthly timestamps following the last timestamp
    last = pd.to_datetime(info["last_timestamp"]) + pd.offsets.MonthBegin(1)
    idx = pd.date_range(start=last, periods=horizon, freq="MS")
    out = [
        {"date": str(dt.date()), "forecast": float(max(0.0, y))} for dt, y in zip(idx, yhat)
    ]
    return _sanitize_for_json({
        "barangay": barangay,
        "crime_type": crime_type,
        "horizon_months": horizon,
        "values": out,
        "metrics": info.get("metrics", {}),
    })


@app.get("/chart")
def chart(
    barangay: str = Query(..., description="Barangay name, e.g., 'Barangay 41'"),
    crime_type: str = Query(..., description="Crime type, e.g., 'Theft'"),
    months: Optional[int] = Query(6, ge=1, le=24, description="Forecast horizon in months"),
):
    # Load history from CSV
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    sub = df[(df["barangay"] == barangay) & (df["crime_type"] == crime_type)]
    if sub.empty:
        raise HTTPException(status_code=404, detail="No historical rows for the requested series.")
    history = [
        {"date": str(d.date()), "count": int(c)} for d, c in zip(sub["date"], sub["count"]) 
    ]

    # Forecast using existing endpoint logic
    fc = forecast(barangay=barangay, crime_type=crime_type, months=months)
    return _sanitize_for_json({
        "barangay": barangay,
        "crime_type": crime_type,
        "history": history,
        "forecast": fc["values"],
        "metrics": fc.get("metrics", {}),
    })


