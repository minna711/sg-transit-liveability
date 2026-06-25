# 🏙️ SG Transit Liveability Index

> Real-time neighbourhood evaluation for Singapore flat hunters — combining live taxi availability, bus frequency, HDB resale prices and ML predictions into a single **District Connectivity Score**.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.9-orange)

---

## 🎯 Problem Statement

> **"How might we help Singapore flat buyers make a confident decision about non-MRT estates by combining real-time transport availability with historical resale prices into a single liveability score?"**

Singapore has 55+ HDB towns — but not all are MRT-connected. If you're considering moving to a non-MRT estate like **Marine Parade**, **Tengah**, or **Punggol**, how do you know if the bus and taxi connectivity is good enough on a typical Tuesday morning at 7:30am?

This pipeline answers that question with **real-time data + ML predictions**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                          │
│  LTA DataMall API          data.gov.sg        OneMap API    │
│  Taxi every 60s        HDB 233k records     Geocoding+MRT   │
│  Bus every 3 min         (2017–2026)        Routing+Search  │
└──────────────┬──────────────────┬────────────────┬──────────┘
               ▼                  ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                       STORAGE                               │
│     SQLite (transport.db)          DuckDB (hdb.duckdb)      │
│  taxi_snapshots, predictions,    stg_hdb_raw, geo_cache     │
│  anomaly_alerts, model_metrics   9,712 geocoded HDB blocks  │
└──────────────┬──────────────────┬────────────────────────────┘
               ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING + ANALYTICS                   │
│  Taxi Disappearance Engine     HDB Price Analytics          │
│  SVY21 + 20m spatial buffer    Town summaries, VFM score    │
│  Bus headway extraction        6-month price forecast       │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML LAYER                               │
│  Ridge Regression Forecaster   HourlyForecaster (24hr)      │
│  +30/+60/+120 min predictions  Peak Hour Ratings            │
│  Anomaly Detection             Day Pattern Heatmap          │
│  HDB Price Forecast (Prophet)  APScheduler / Airflow DAG    │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVING                                │
│   FastAPI REST API          Streamlit Dashboard (3 pages)   │
│   /evaluate /rank           Live charts, 24hr forecast      │
│   /forecast/24h             Peak hour ratings               │
│   /forecast/peaks           Day pattern heatmap             │
│   /forecast/price           HDB map + postal code search    │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🚕 Real-time Transport
- **Taxi Disappearance Engine** — detects estimated pickups using 20m spatial buffers (SVY21 projection)
- **Bus Reliability Factor** — average headway with >120min anomaly filter
- **District Connectivity Score (0–100)** — `(Bus×0.5) + (Taxi×0.3) − (Friction×0.2)`
- **Anomaly Alerts** — LOW_TAXI, HIGH_FLUX, BUS_GAP

### 🤖 ML Forecasting (4 models)
| Model | Output | Horizon |
|-------|--------|---------|
| Ridge Regression | Taxi count | +30/60/120 min |
| HourlyForecaster | 24hr chart | Next 24 hours |
| PeakHourPredictor | 🟢🟡🔴 ratings | Tomorrow's peaks |
| DayPatternAnalyser | 7×24 heatmap | Historical pattern |
| HDBPriceForecaster | Price trend | Next 6 months |

### 🏠 HDB Resale Integration
- 233,479 transactions (2017–2026) from data.gov.sg
- 9,712 HDB blocks geocoded via OneMap (100% hit rate)
- Value-for-Money score with **dynamic weight sliders**
- 6-month price forecast using Prophet/linear regression

### 📍 Postal Code Search
Enter any Singapore postal code to see:
- 🚇 Nearest MRT + walking distance
- 🚌 Live bus arrivals at nearby stops (with load: Seated/Standing/Crowded)
- 🚕 Live taxi count in area
- 💰 Nearby HDB avg resale price
- ⏱️ Commute time to CBD by public transport
- 📊 Connectivity score for that location

### 🗺️ Singapore Map
- HDB price heatmap (🔴 expensive → 🟢 affordable)
- All 6 MRT lines overlaid with line colours
- District boundary boxes
- Live taxi density heatmap

### ⚙️ MLOps
- Daily model retraining at 08:00 SGT
- Prediction evaluation vs actuals
- Apache Airflow DAG with task dependencies + auto-retry
- Docker Compose for one-command deployment

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)
```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up
```

Access:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- Airflow: http://localhost:8080 (admin/admin)

### Option 2 — Local
```bash
# Setup
uv venv && .venv/Scripts/activate  # Windows
uv pip install -r requirements.txt

# API keys (Windows Environment Variables)
# LTA_API_KEY = your LTA DataMall key
# ONEMAP_TOKEN = your OneMap token

# Seed + run
python main.py --seed
python main.py

# Dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
sg-transit-liveability/
├── config.py                   ← frozen dataclass singleton
├── main.py                     ← orchestrator + CLI
├── api.py                      ← FastAPI endpoints
├── requirements.txt
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── .env.example
│
├── ingestion/
│   ├── client.py               ← LTA HTTP client (v3 API)
│   └── workers.py              ← TaxiWorker + BusWorker (RLock)
│
├── processing/
│   ├── taxi.py                 ← Disappearance engine (SVY21)
│   └── spatial.py              ← Bbox filters
│
├── analytics/
│   └── engine.py               ← Connectivity score + CV stability
│
├── ml/
│   ├── forecaster.py           ← Ridge regression (+30/60/120 min)
│   ├── extended_forecaster.py  ← 24hr, peak hours, day pattern, HDB price
│   ├── anomaly.py              ← Anomaly detection
│   └── batch_jobs.py           ← APScheduler jobs
│
├── storage/
│   └── database.py             ← SQLite persistence
│
├── hdb/
│   ├── geocoder.py             ← OneMap geocoding (100% hit rate)
│   ├── analytics.py            ← DuckDB price queries + VFM
│   ├── map_page.py             ← Streamlit map + postal code search
│   ├── onemap_services.py      ← Nearest MRT, bus stops, routing
│   ├── planning_areas.py       ← All 55 Singapore planning areas
│   └── quality_check.py        ← Data quality report + auto-fix
│
├── dashboard/
│   └── app.py                  ← 3-page Streamlit dashboard
│
└── airflow/
    └── dags/
        └── sg_transit_pipeline.py  ← Airflow DAG (daily + 30min)
```

---

## 🔌 API Endpoints

```
GET /evaluate?min_lon=&max_lon=&min_lat=&max_lat=  → connectivity score
GET /rank                                           → all districts leaderboard
GET /predictions/{district}                         → ML forecasts
GET /alerts?district=                               → anomaly alerts
GET /forecast/24h/{district}                        → 24hr hourly forecast
GET /forecast/peaks/{district}                      → peak hour ratings
GET /forecast/pattern/{district}                    → day×hour heatmap
GET /forecast/price/{town}                          → HDB price forecast
GET /health                                         → liveness check
```

---

## 📊 Score Formula

```
Connectivity Score (0–100) =
  (Bus Frequency Score  × 50%)
+ (Taxi Stability Score × 30%)
− (Friction Penalty     × 20%)
```

| Score | Verdict |
|-------|---------|
| 75–100 | ✅ Well connected |
| 50–74  | ⚠️ Moderate |
| 0–49   | ❌ Poor connectivity |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14 |
| Geospatial | GeoPandas, Shapely, PyProj (SVY21) |
| ML | scikit-learn, Prophet |
| Storage | SQLite, DuckDB |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Scheduling | APScheduler + Apache Airflow |
| Containers | Docker, Docker Compose |
| Data sources | LTA DataMall, data.gov.sg, OneMap |

---

## 🔜 Roadmap

- [ ] Wireframe redesign (designer in progress)
- [ ] MLflow experiment tracking
- [ ] Git LFS for hdb.duckdb
- [ ] All 55 planning areas in dashboard dropdown
- [ ] dbt transforms for HDB analytics
- [ ] Presentation video recording

---

## 📄 License

MIT — data under [Singapore Open Data Licence](https://datamall.lta.gov.sg)

*SIT SNAIC Data Engineering Project 2026*
