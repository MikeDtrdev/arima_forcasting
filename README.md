## E-Responde Crime Forecasting (ARIMA + FastAPI)

This project creates a simple, reproducible pipeline to forecast monthly crime incidents for Barangay 41 and Barangay 43 (Tondo, Manila) using ARIMA, and serves forecasts via FastAPI.

### What you get
- Synthetic monthly dataset for 2019–2024 covering 9 crime types
- ARIMA training per barangay and crime type with metrics and saved models
- FastAPI service to request future forecasts in real time

### Quick start (Windows PowerShell)
```powershell
cd "C:\Users\Miked\arima forcasting"
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# 1) Generate data (2019-01 to 2024-12)
python scripts/generate_data.py --out data/crimes_2019_2024.csv

# 2) Train ARIMA models and build registry
python scripts/train_arima.py --data data/crimes_2019_2024.csv --models_dir models

# 3) Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Try it
- Open `http://127.0.0.1:8000/docs` to explore the API.
- Example: `GET /forecast?barangay=Barangay%2043&crime_type=Theft&months=6`.

### Project structure
```
scripts/            # data generation + training
app/                # FastAPI service
data/               # generated CSV (ignored by Git)
models/             # saved models + registry.json (ignored by Git)
```

### Notes
- Replace the synthetic dataset with real incident data when available; keep the same columns.
- ARIMA models are seasonal (m=12). Edit `scripts/train_arima.py` if you want different settings.

