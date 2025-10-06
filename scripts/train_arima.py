import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional dependency: pmdarima for auto_arima. Provide graceful fallback.
try:
    from pmdarima import auto_arima  # type: ignore
    _HAS_PM = True
except Exception:  # pragma: no cover - optional path
    auto_arima = None  # type: ignore
    _HAS_PM = False
    from statsmodels.tsa.statespace.sarimax import SARIMAX


def prepare_series(df: pd.DataFrame, barangay: str, crime_type: str) -> pd.Series:
    sub = (
        df[(df["barangay"] == barangay) & (df["crime_type"] == crime_type)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    # Aggregate daily data to monthly if needed
    if len(sub) > 72:  # More than 6 years of monthly data
        sub["month"] = sub["date"].dt.to_period("M").dt.to_timestamp()
        sub = sub.groupby("month")["count"].sum().reset_index()
        sub = sub.rename(columns={"month": "date"})
    
    idx = pd.PeriodIndex(sub["date"], freq="M").to_timestamp()
    y = pd.Series(sub["count"].values, index=idx)
    y = y.asfreq("MS")
    # Fill any missing months with zeros for stability
    y = y.fillna(0.0).astype(float)
    return y


def fit_model(y: pd.Series):
    """Fit a seasonal monthly ARIMA model.

    Uses pmdarima.auto_arima when available; otherwise performs a small grid
    search with statsmodels SARIMAX and selects the best AIC.
    """
    if _HAS_PM and auto_arima is not None:
        model = auto_arima(
            y,
            seasonal=True,
            m=12,
            stepwise=True,
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_order=None,
            information_criterion="aicc",
        )
        return model

    # Fallback: compact grid over (p,d,q)x(P,D,Q)m with m=12
    best_aic = float("inf")
    best_res = None
    candidates = [(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1, 2)]
    seasonal = [(P, D, Q, 12) for P in (0, 1) for D in (0, 1) for Q in (0, 1)]

    for (p, d, q) in candidates:
        for (P, D, Q, m) in seasonal:
            try:
                mdl = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if mdl.aic < best_aic:
                    best_aic = mdl.aic
                    best_res = mdl
            except Exception:
                continue

    if best_res is None:
        # Try Holt-Winters as robust fallback
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            hw = ExponentialSmoothing(
                y,
                trend=None,
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit()
            return hw
        except Exception:
            # Last resort: simple non-seasonal model
            best_res = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
    return best_res


def fit_hw_model(y: pd.Series):
    """Fast, robust Holt-Winters seasonal model (additive)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    return ExponentialSmoothing(
        y,
        trend=None,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()


def backtest_mae_rmse(y: pd.Series, model, horizon: int = 3) -> Tuple[float, float]:
    # Simple expanding-window backtest
    n = len(y)
    splits = []
    start = max(18, n // 2)
    for t in range(start, n - horizon + 1):
        splits.append((t, t + horizon))

    all_true = []
    all_pred = []
    for train_end, test_end in splits:
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:test_end]
        try:
            m = fit_model(y_train)
            steps = len(y_test)
            try:
                fc = m.predict(n_periods=steps)  # type: ignore[attr-defined]
            except Exception:
                try:
                    fc = m.forecast(steps=steps)  # type: ignore[attr-defined]
                except Exception:
                    fc = m.get_forecast(steps=steps).predicted_mean  # type: ignore[attr-defined]
            all_true.extend(y_test.values)
            all_pred.extend(np.asarray(fc))
        except Exception:
            # Skip this split if model fitting fails
            continue

    if not all_true:
        return float("nan"), float("nan")

    mae = float(mean_absolute_error(all_true, all_pred))
    rmse = float(np.sqrt(mean_squared_error(all_true, all_pred)))
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Train ARIMA models per barangay and crime type")
    parser.add_argument("--data", required=True, help="CSV with columns: date, barangay, crime_type, count")
    parser.add_argument("--models_dir", required=True, help="Directory to save models and registry.json")
    parser.add_argument("--forecast_months", type=int, default=6)
    parser.add_argument("--fast", action="store_true", help="Use fast Holt-Winters model and skip backtesting for speed")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])
    barangays = sorted(df["barangay"].unique())
    crimes = sorted(df["crime_type"].unique())

    registry: Dict[str, Dict] = {
        "forecast_months_default": args.forecast_months,
        "series": {},
    }

    for b in barangays:
        for c in crimes:
            key = f"{b}__{c}"
            y = prepare_series(df, b, c)
            if args.fast:
                # Skip heavy backtesting in fast mode
                mae, rmse = float("nan"), float("nan")
                try:
                    model = fit_hw_model(y)
                except Exception:
                    # Simple seasonal naive as final fallback
                    class NaiveSeasonalModel:
                        def __init__(self, values: pd.Series, m: int = 12):
                            self.values = values
                            self.m = m

                        def predict(self, n_periods: int):
                            if len(self.values) < self.m:
                                base = np.repeat(np.mean(self.values), n_periods)
                                return base
                            pattern = self.values.iloc[-self.m :].values
                            reps = int(np.ceil(n_periods / self.m))
                            return np.tile(pattern, reps)[:n_periods]

                        def forecast(self, steps: int):
                            return self.predict(steps)

                    model = NaiveSeasonalModel(y)
            else:
                try:
                    mae, rmse = backtest_mae_rmse(y, None, horizon=3)
                except Exception:
                    mae, rmse = float("nan"), float("nan")
                try:
                    model = fit_model(y)
                except Exception:
                    model = fit_hw_model(y)

            model_path = models_dir / f"model_{hash(key)}.pkl"
            import joblib

            joblib.dump(model, model_path)

            last_date = y.index.max()
            registry["series"][key] = {
                "barangay": b,
                "crime_type": c,
                "n_obs": int(len(y)),
                "last_timestamp": str(last_date.date()),
                "model_path": str(model_path),
                "metrics": {"mae": mae, "rmse": rmse},
            }

    with open(models_dir / "registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved {len(registry['series'])} models to {models_dir}")


if __name__ == "__main__":
    main()


