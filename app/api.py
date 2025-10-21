from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import re
from datetime import datetime, timedelta
# Firebase Admin imports and initialization
try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS")
    FIREBASE_DB_URL = os.environ.get("FIREBASE_DB_URL")
    if FIREBASE_DB_URL and not firebase_admin._apps:
        cred = None
        if FIREBASE_CREDENTIALS:
            # Support either path to JSON file or raw JSON string
            if os.path.isfile(FIREBASE_CREDENTIALS):
                cred = credentials.Certificate(FIREBASE_CREDENTIALS)
            else:
                try:
                    cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS))
                except Exception:
                    cred = None
        if cred:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DB_URL
            })
except Exception as _fb_err:
    # Firebase optional; API should still run without it
    print(f"Firebase init warning: {_fb_err}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to models directory
MODELS_DIR = Path("../tondo_forecasts/models")
FORECASTS_FILE = Path("../tondo_forecasts/tondo_crime_forecasts.json")
HISTORICAL_DATA_FILE = Path("../tondo_crime_data_barangay_41_43_2019_2025.csv")

# Cache for loaded models to improve performance
model_cache = {}

# Robust months parser to accept values like "12", "12 Months", or empty
def parse_months(value):
    if value is None or value == "":
        return 12
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group())
        return 12

# Generate forecast robustly across model types (statsmodels, pmdarima)
def generate_forecast(model, months):
    try:
        if hasattr(model, 'forecast'):
            yhat = model.forecast(steps=months)
        elif hasattr(model, 'predict'):
            try:
                yhat = model.predict(n_periods=months)
            except TypeError:
                yhat = model.predict(steps=months)
        elif hasattr(model, 'get_forecast'):
            res = model.get_forecast(steps=months)
            yhat = getattr(res, 'predicted_mean', res)
        else:
            raise AttributeError('Model does not support forecasting')
        arr = np.asarray(yhat, dtype=float)
        if np.isnan(arr).any():
            for i in range(arr.size):
                if np.isnan(arr[i]):
                    arr[i] = arr[i-1] if i > 0 and not np.isnan(arr[i-1]) else 0.0
        arr = np.clip(arr, 0, None)
        return arr.tolist()
    except Exception as e:
        print(f"Forecast generation error: {e}")
        return [0.0] * int(months)

# Add Firebase monthly aggregation helper
def fetch_monthly_counts_from_firebase(crime_type, location):
    labels, values = [], []
    try:
        if 'firebase_admin' in globals() and FIREBASE_DB_URL and firebase_admin._apps:
            ref = firebase_db.reference('civilian_crime_reports')
            snapshot = ref.order_by_child('timestamp').get() or {}
            rows = []
            for _, val in snapshot.items():
                ct = val.get('crime_type') or val.get('crimeType')
                loc = val.get('location')
                ts = val.get('timestamp')
                if ct == crime_type and loc == location and isinstance(ts, (int, float)):
                    # Firebase timestamps are ms since epoch; convert to month string
                    dt = datetime.utcfromtimestamp(ts/1000.0)
                    rows.append({"month": dt.strftime("%Y-%m")})
            if rows:
                df = pd.DataFrame(rows)
                grouped = df.groupby('month').size().reset_index(name='count').sort_values('month')
                labels = grouped['month'].tolist()
                values = grouped['count'].astype(int).tolist()
    except Exception as e:
        print(f"Firebase monthly aggregation error: {e}")
    return labels, values

def load_model(crime_type, location):
    """Load a model from disk or cache with directory fallbacks"""
    key = f"{crime_type}__{location}"
    filename = f"model_{key.replace(' ', '_').replace('/', '_')}.pkl"
    # Candidate directories for robustness
    candidates = [
        Path("../tondo_forecasts/models") / filename,
        Path("./tondo_forecasts/models") / filename,
        Path("../tondo_crime_api/tondo_forecasts/models") / filename,
    ]
    if key in model_cache:
        return model_cache[key]
    for model_path in candidates:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    model_cache[key] = model
                    return model
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                return None
    return None

# Helper: detect degenerate statsmodels fits (extreme params)
def is_degenerate_model(model):
    try:
        params = getattr(model, 'params', None)
        if params is not None:
            return bool((np.abs(np.asarray(params, dtype=float)) > 1e6).any())
    except Exception:
        pass
    return False

# Helper: dynamic ARIMA forecast on provided monthly count series
def dynamic_arima_forecast(monthly_counts, months):
    try:
        import pmdarima as pm
        y = np.array(monthly_counts, dtype=float)
        if y.size < 6:
            return None
        seasonal_flag = True if y.size >= 24 else False
        auto_model = pm.auto_arima(
            y,
            seasonal=seasonal_flag,
            m=12 if seasonal_flag else 1,
            suppress_warnings=True,
            error_action='ignore',
            enforce_stationarity=True,
            enforce_invertibility=True,
            boxcox=True
        )
        fc = auto_model.predict(n_periods=int(months)).tolist()
        return [max(0.0, float(v)) for v in fc]
    except Exception as e:
        print(f"Dynamic ARIMA error (pmdarima): {e}")
        # Fallback: use statsmodels SARIMAX grid search
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y = np.array(monthly_counts, dtype=float)
            if y.size < 6:
                return None
            # Compact but effective grid
            p_vals = [0, 1]
            d_vals = [0, 1]
            q_vals = [0, 1]
            P_vals = [0, 1]
            D_vals = [1]
            Q_vals = [0, 1]
            m = 12
            best_aic = np.inf
            best_fit = None
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
                                    except Exception:
                                        continue
            if best_fit is None:
                # Fallback simple seasonal model
                model = SARIMAX(
                    y,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    trend='c',
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                best_fit = model.fit(disp=False)
            fc = best_fit.forecast(steps=int(months))
            arr = np.asarray(fc, dtype=float)
            arr = np.clip(arr, 0, None)
            return arr.tolist()
        except Exception as e2:
            print(f"Dynamic ARIMA error (statsmodels fallback): {e2}")
            return None
    except Exception as e:
        print(f"Dynamic ARIMA error: {e}")
        return None

# Naive baseline forecast when history is too short or ARIMA fails
def naive_baseline_forecast(monthly_counts, months):
    y = np.array(monthly_counts, dtype=float)
    if y.size == 0:
        return None
    if y.size >= 12:
        base = float(np.mean(y[-12:]))
    else:
        base = float(np.mean(y))
    base = max(0.0, base)
    return [base] * int(months)

# CSV monthly aggregation helper
def fetch_monthly_counts_from_csv(crime_type, location):
    labels, values = [], []
    try:
        if HISTORICAL_DATA_FILE.exists():
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            df.columns = [c.strip() for c in df.columns]
            if 'crimeType' in df.columns:
                df['crimeType'] = df['crimeType'].astype(str).str.strip()
            if 'location' in df.columns:
                df['location'] = df['location'].astype(str).str.strip()
            df = df[(df['crimeType'] == crime_type.strip()) & (df['location'] == location.strip())]
            if not df.empty:
                df['month'] = pd.to_datetime(df['dateTime'], errors='coerce').dt.to_period('M').astype(str)
                grouped = df.groupby('month').size().reset_index(name='count').sort_values('month')
                labels = grouped['month'].tolist()
                values = grouped['count'].astype(int).tolist()
    except Exception as e:
        print(f"CSV aggregation error: {e}")
    return labels, values

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Generate a forecast for a specific crime type and location"""
    crime_type = request.args.get('crime_type')
    location = request.args.get('location')
    months = parse_months(request.args.get('months', 12))
    
    if not crime_type or not location:
        return jsonify({"error": "Missing crime_type or location parameter"}), 400
    
    # Prefer to build history from Firebase if configured
    fb_labels, fb_values = [], []
    try:
        if 'firebase_admin' in globals() and firebase_admin._apps:
            ref = firebase_db.reference('civilian_crime_reports')
            snapshot = ref.get() or {}
            monthly = {}
            for _, val in snapshot.items():
                if not isinstance(val, dict):
                    continue
                ct = val.get('crime_type') or val.get('crimeType')
                loc = val.get('location')
                if ct != crime_type or loc != location:
                    continue
                ts = val.get('timestamp')
                dt = None
                if isinstance(ts, (int, float)) and ts > 0:
                    dt = datetime.utcfromtimestamp(float(ts)/1000.0)
                elif isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts)
                    except Exception:
                        dt = None
                if not dt:
                    continue
                mkey = dt.strftime('%Y-%m')
                monthly[mkey] = monthly.get(mkey, 0) + 1
            for m in sorted(monthly.keys()):
                fb_labels.append(m)
                fb_values.append(int(monthly[m]))
    except Exception as e:
        print(f"Firebase history error: {e}")
    
    model = load_model(crime_type, location)
    if model is None and not fb_values:
        return jsonify({"error": f"Model not found for {crime_type} at {location} and no Firebase history"}), 404
    
    try:
        forecast_values = None
        # Prefer dynamic ARIMA if we have Firebase monthly counts
        if fb_values and len(fb_values) >= 6:
            forecast_values = dynamic_arima_forecast(fb_values, months)
            if forecast_values is None:
                forecast_values = naive_baseline_forecast(fb_values, months)
        # Fallback to pre-trained model
        if forecast_values is None and model is not None:
            forecast_values = generate_forecast(model, months)
            if (all(v == 0.0 for v in forecast_values)) or is_degenerate_model(model):
                csv_labels, csv_values = fetch_monthly_counts_from_csv(crime_type, location)
                if csv_values:
                    alt_fc = dynamic_arima_forecast(csv_values, months)
                    if alt_fc is None:
                        alt_fc = naive_baseline_forecast(csv_values, months)
                    if alt_fc is not None:
                        forecast_values = alt_fc
        if forecast_values is None:
            csv_labels, csv_values = fetch_monthly_counts_from_csv(crime_type, location)
            baseline = naive_baseline_forecast(csv_values, months) if csv_values else None
            forecast_values = baseline if baseline is not None else [0.0] * int(months)
        
        # Determine forecast starting month based on Firebase or CSV history
        last_month_dt = None
        if fb_labels:
            try:
                last_month_dt = datetime.strptime(fb_labels[-1], "%Y-%m")
            except Exception:
                last_month_dt = None
        if last_month_dt is None:
            csv_labels, csv_values = fetch_monthly_counts_from_csv(crime_type, location)
            if csv_labels:
                try:
                    last_month_dt = datetime.strptime(csv_labels[-1], "%Y-%m")
                except Exception:
                    last_month_dt = None
        if last_month_dt is None:
            today = datetime.now()
            last_month_dt = datetime(today.year, today.month, 1)
        
        forecast_dates = []
        for i in range(1, months + 1):
            next_month = last_month_dt + timedelta(days=32 * i)
            next_month = datetime(next_month.year, next_month.month, 1)
            forecast_dates.append(next_month.strftime("%Y-%m-%d"))
        
        return jsonify({
            "crime_type": crime_type,
            "location": location,
            "forecast": {
                "dates": forecast_dates,
                "values": forecast_values
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error generating forecast: {str(e)}"}), 500

@app.route('/api/crime_types', methods=['GET'])
def get_crime_types():
    """Get all available crime types"""
    crime_types = set()
    
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            parts = filename[6:-4].split("__")
            if len(parts) >= 1:
                crime_type = parts[0].replace("_", " ")
                crime_types.add(crime_type)
    
    return jsonify({"crime_types": sorted(list(crime_types))})

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get allowed locations (restricted to Barangay 41 and Barangay 43)"""
    allowed_locations = {"Barangay 41", "Barangay 43"}
    return jsonify({"locations": sorted(list(allowed_locations))})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models"""
    models = []
    
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("model_") and filename.endswith(".pkl"):
            parts = filename[6:-4].split("__")
            if len(parts) >= 2:
                crime_type = parts[0].replace("_", " ")
                location = parts[1].replace("_", " ")
                models.append({
                    "crime_type": crime_type,
                    "location": location,
                    "model_file": filename
                })
    
    return jsonify({"models": models})

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary information about the forecasts"""
    try:
        with open(FORECASTS_FILE, 'r') as f:
            data = json.load(f)
        
        # Extract summary information
        crime_types = set()
        locations = set()
        
        for key in data.get("forecasts", {}).keys():
            parts = key.split("__")
            if len(parts) >= 2:
                crime_types.add(parts[0])
                locations.add(parts[1])
        
        summary = {
            "total_models": len(data.get("model_files", {})),
            "crime_types": sorted(list(crime_types)),
            "locations": sorted(list(locations)),
            "timestamp": data.get("timestamp")
        }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Error loading summary: {str(e)}"}), 500

@app.route('/api/all_forecasts', methods=['GET'])
def get_all_forecasts():
    """Get all pre-computed forecasts"""
    try:
        with open(FORECASTS_FILE, 'r') as f:
            data = json.load(f)
        
        return jsonify({"forecasts": data.get("forecasts", {})})
    except Exception as e:
        return jsonify({"error": f"Error loading forecasts: {str(e)}"}), 500

@app.route('/api/report', methods=['POST'])
def create_report():
    """Accept a civilian crime report and push to Firebase Realtime Database.
    Expects JSON body with keys like: crime_type, location, severity, description, reporter_id, latitude, longitude, timestamp(optional).
    """
    try:
        if 'firebase_admin' not in globals() or not FIREBASE_DB_URL or not firebase_admin._apps:
            return jsonify({"error": "Firebase is not configured. Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL environment variables."}), 500
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        # Basic validation
        crime_type = data.get('crime_type') or data.get('crimeType')
        location = data.get('location')
        if not crime_type or not location:
            return jsonify({"error": "Missing required fields: crime_type and location"}), 400
        # Prepare payload
        payload = {
            "crime_type": crime_type,
            "location": location,
            "severity": data.get('severity'),
            "description": data.get('description'),
            "reporter_id": data.get('reporter_id'),
            "latitude": data.get('latitude'),
            "longitude": data.get('longitude'),
            # Use server timestamp if client didn't provide
            "timestamp": data.get('timestamp') or {".sv": "timestamp"},
            "source": data.get('source') or "civilian_app"
        }
        ref = firebase_db.reference('civilian_crime_reports')
        new_ref = ref.push(payload)
        return jsonify({"status": "success", "id": new_ref.key})
    except Exception as e:
        return jsonify({"error": f"Failed to create report: {str(e)}"}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Fetch recent civilian crime reports from Firebase Realtime Database.
    Query params: limit (default 50)
    """
    try:
        if 'firebase_admin' not in globals() or not FIREBASE_DB_URL or not firebase_admin._apps:
            return jsonify({"error": "Firebase is not configured. Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL environment variables."}), 500
        limit = int(request.args.get('limit', 50))
        ref = firebase_db.reference('civilian_crime_reports')
        # Order by timestamp if present
        snapshot = ref.order_by_child('timestamp').limit_to_last(limit).get() or {}
        # Convert to list sorted by timestamp
        items = []
        for key, val in snapshot.items():
            val = val or {}
            val['id'] = key
            items.append(val)
        items.sort(key=lambda x: x.get('timestamp', 0))
        return jsonify({"reports": items, "count": len(items)})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch reports: {str(e)}"}), 500

@app.route('/api/visualization', methods=['GET'])
def get_visualization_data():
    """Get data formatted for visualization charts with historical + dashed forecast indicator"""
    crime_type = request.args.get('crime_type')
    location = request.args.get('location')
    months = parse_months(request.args.get('months', 12))
    if not crime_type or not location:
        return jsonify({"error": "Missing crime_type or location parameter"}), 400
    model = load_model(crime_type, location)
    if model is None:
        pass
    try:
        # Build historical monthly counts by merging CSV base with Firebase increments
        csv_labels, csv_values = fetch_monthly_counts_from_csv(crime_type, location)
        fb_labels, fb_values = fetch_monthly_counts_from_firebase(crime_type, location)
        monthly_map = {}
        for lbl, val in zip(csv_labels, csv_values):
            monthly_map[lbl] = monthly_map.get(lbl, 0) + int(val)
        for lbl, val in zip(fb_labels, fb_values):
            monthly_map[lbl] = monthly_map.get(lbl, 0) + int(val)
        history_labels = sorted(monthly_map.keys())
        history_values = [int(monthly_map[lbl]) for lbl in history_labels]
        # Determine forecast starting month based on history
        if history_labels:
            last_month_dt = datetime.strptime(history_labels[-1], "%Y-%m")
        else:
            today = datetime.now()
            last_month_dt = datetime(today.year, today.month, 1)
        # Generate forecast and future month labels starting next month
        forecast_values = []
        if model is not None:
            forecast_values = generate_forecast(model, months)
            # Fallback to dynamic fit if degenerate
            if (all(v == 0.0 for v in forecast_values)) or is_degenerate_model(model):
                base_series = history_values if history_values else []
                alt_fc = dynamic_arima_forecast(base_series, months) if base_series else None
                if alt_fc is None and base_series:
                    alt_fc = naive_baseline_forecast(base_series, months)
                if alt_fc is not None:
                    forecast_values = alt_fc
        else:
            alt_fc = dynamic_arima_forecast(history_values, months) if history_values else None
            if alt_fc is None and history_values:
                alt_fc = naive_baseline_forecast(history_values, months)
            forecast_values = alt_fc if alt_fc is not None else [None] * months
        forecast_labels = []
        start_dt = last_month_dt
        for i in range(1, months + 1):
            next_month = start_dt + timedelta(days=32 * i)
            next_month = datetime(next_month.year, next_month.month, 1)
            forecast_labels.append(next_month.strftime("%Y-%m"))
        all_labels = history_labels + forecast_labels
        historical_dataset_values = history_values + [None] * months
        forecast_dataset_values = [None] * len(history_labels) + (
            forecast_values if forecast_values is not None else [None] * months
        )
        chart_data = {
            "type": "line",
            "data": {
                "labels": all_labels,
                "datasets": [
                    {
                        "label": "Historical",
                        "data": historical_dataset_values,
                        "borderColor": "rgb(54, 162, 235)",
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "tension": 0.1,
                        "fill": False
                    },
                    {
                        "label": "Forecast",
                        "data": forecast_dataset_values,
                        "borderColor": "rgb(255, 99, 132)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "tension": 0.1,
                        "fill": False,
                        "borderDash": [8, 6]
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": "Crime Count"}},
                    "x": {"title": {"display": True, "text": "Month"}}
                },
                "plugins": {"title": {"display": True, "text": f"Crime Forecast (dashed): {crime_type} at {location}"}}
            }
        }
        return jsonify({
            "crime_type": crime_type,
            "location": location,
            "chart_config": chart_data,
            "raw_data": {"history": {"labels": history_labels, "values": history_values}, "forecast": {"labels": forecast_labels, "values": forecast_values}}
        })
    except Exception as e:
        return jsonify({"error": f"Error generating visualization: {str(e)}"}), 500

@app.route('/')
def index():
    """API documentation"""
    return """
    <html>
        <head>
            <title>Tondo Crime Forecast API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1 { color: #333; }
                h2 { color: #555; margin-top: 30px; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
                pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Tondo Crime Forecast API</h1>
            <p>This API provides access to ARIMA forecasting models for crime prediction in the Tondo area.</p>
            
            <h2>Endpoints</h2>
            
            <h3>1. Generate Forecast</h3>
            <p><code>GET /api/forecast?crime_type={crime_type}&location={location}&months={months}</code></p>
            <p>Generate a forecast for a specific crime type and location.</p>
            <p>Parameters:</p>
            <ul>
                <li><code>crime_type</code> (required): Type of crime</li>
                <li><code>location</code> (required): Location in Tondo</li>
                <li><code>months</code> (optional): Number of months to forecast (default: 12)</li>
            </ul>
            
            <h3>2. Get Crime Types</h3>
            <p><code>GET /api/crime_types</code></p>
            <p>Get a list of all available crime types.</p>
            
            <h3>3. Get Locations</h3>
            <p><code>GET /api/locations</code></p>
            <p>Get a list of all available locations.</p>
            
            <h3>4. Get Models</h3>
            <p><code>GET /api/models</code></p>
            <p>Get a list of all available models.</p>
            
            <h3>5. Get Summary</h3>
            <p><code>GET /api/summary</code></p>
            <p>Get summary information about the forecasts.</p>
            
            <h3>6. Get All Forecasts</h3>
            <p><code>GET /api/all_forecasts</code></p>
            <p>Get all pre-computed forecasts.</p>
            
            <h3>7. Test Interface</h3>
            <p><a href="/test_api.html">Interactive API Test Page</a></p>
        </body>
    </html>
    """

@app.route('/test_api.html')
def serve_test_page():
    """Serve the test API HTML page"""
    return send_from_directory('.', 'test_api.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)