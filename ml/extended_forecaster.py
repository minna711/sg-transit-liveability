"""
ml/extended_forecaster.py
=========================
Extended ML predictions beyond the basic +30/60/120 min Ridge model.

1. 24-hour hourly taxi forecast    — Ridge model extended to 24 horizons
2. Peak hour prediction            — classify 7-9am as Good/Moderate/Poor
3. Day of week pattern             — historical avg by hour + day (heatmap data)
4. HDB price forecast              — Prophet time series on 9yr transaction data

MLOps hooks match the main forecaster:
  train()    → fit models, save to disk
  predict()  → load + generate forecasts
  evaluate() → compare vs actuals
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone, timedelta

SGT = timezone(timedelta(hours=8))
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from config import cfg
from storage.database import fetch_snapshots, insert_prediction, insert_model_metrics

log = logging.getLogger(__name__)

MODEL_DIR = Path(cfg.model_dir)

# 24 hourly horizons
HOURLY_HORIZONS = list(range(30, 24 * 60 + 1, 60))  # 30,90,150...1410 min

FEATURE_COLS = [
    "hour", "minute", "weekday",
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "roll5_mean", "roll15_mean", "roll5_std",
]


# ── Feature engineering ────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df = df.set_index("fetched_at").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["hour"]    = df.index.hour
    df["minute"]  = df.index.minute
    df["weekday"] = df.index.weekday
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["taxi_count"].shift(lag)
    df["roll5_mean"]  = df["taxi_count"].rolling(5,  min_periods=1).mean()
    df["roll15_mean"] = df["taxi_count"].rolling(15, min_periods=1).mean()
    df["roll5_std"]   = df["taxi_count"].rolling(5,  min_periods=1).std().fillna(0)
    return df.dropna()


# ── 1. 24-Hour Hourly Taxi Forecast ───────────────────────────────────────────

class HourlyForecaster:
    """
    Extends Ridge regression to predict taxi count every hour for next 24 hours.
    Returns a DataFrame with columns: hour_ahead, predicted_at, predicted_count
    """

    def __init__(self, district: str):
        self.district  = district
        self._models: dict[int, Pipeline] = {}

    def train(self, lookback_min: int = 10080) -> dict:
        rows = fetch_snapshots(self.district, minutes=lookback_min)
        if len(rows) < 50:
            log.warning("[%s] Not enough data for hourly forecast", self.district)
            return {}

        df = _build_features(pd.DataFrame(rows))
        results = {}

        for h in HOURLY_HORIZONS:
            df["y"] = df["taxi_count"].shift(-h)
            train   = df.dropna(subset=["y"])
            if len(train) < 20:
                continue
            X, y = train[FEATURE_COLS].values, train["y"].values
            pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
            pipe.fit(X, y)
            mae = mean_absolute_error(y, pipe.predict(X))
            self._models[h] = pipe
            results[h] = round(mae, 2)

        self._save()
        log.info("[%s] Hourly forecaster trained: %d horizons", self.district, len(self._models))
        return results

    def predict_24h(self) -> pd.DataFrame:
        """
        Generate 24-hour ahead predictions.
        Returns DataFrame: timestamp, hour_ahead, predicted_count, confidence
        """
        self._load()
        rows = fetch_snapshots(self.district, minutes=60)
        if not rows:
            return pd.DataFrame()

        df  = _build_features(pd.DataFrame(rows))
        if df.empty:
            return pd.DataFrame()

        row = df[FEATURE_COLS].iloc[[-1]].values
        now = datetime.now(SGT)
        records = []

        for h in HOURLY_HORIZONS:
            if h in self._models:
                pred = max(0.0, float(self._models[h].predict(row)[0]))
            else:
                # EMA fallback
                pred = float(pd.DataFrame(rows)["taxi_count"].ewm(span=10).mean().iloc[-1])

            pred_time = now + timedelta(minutes=h)
            hour_ahead = h // 60

            records.append({
                "predicted_at":    pred_time,
                "hour_ahead":      hour_ahead,
                "predicted_count": round(pred, 1),
                "horizon_min":     h,
            })

        return pd.DataFrame(records)

    def _model_path(self, h: int) -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return MODEL_DIR / f"{self.district}_hourly_{h}min.pkl"

    def _save(self):
        for h, m in self._models.items():
            with open(self._model_path(h), "wb") as f:
                pickle.dump(m, f)

    def _load(self):
        if self._models: return
        for h in HOURLY_HORIZONS:
            p = self._model_path(h)
            if p.exists():
                with open(p, "rb") as f:
                    self._models[h] = pickle.load(f)


# ── 2. Peak Hour Prediction ────────────────────────────────────────────────────

class PeakHourPredictor:
    """
    Classifies upcoming peak hours as Good / Moderate / Poor for taxi availability.
    Uses historical patterns from the DB.

    Output: {
        "7am_tomorrow": {"rating": "Poor", "expected": 12, "vs_avg": -35%},
        "8am_tomorrow": {"rating": "Moderate", "expected": 18, "vs_avg": -10%},
        ...
    }
    """

    PEAK_HOURS = [7, 8, 17, 18, 19]   # morning + evening peak

    def __init__(self, district: str):
        self.district = district

    def predict_peaks(self, days_ahead: int = 1) -> list[dict]:
        """
        Predict taxi availability for peak hours tomorrow.
        Uses historical average for that hour + day of week as baseline.
        """
        rows = fetch_snapshots(self.district, minutes=7 * 24 * 60)
        if not rows:
            return []

        df = pd.DataFrame(rows)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"])
        df["hour"]    = df["fetched_at"].dt.hour
        df["weekday"] = df["fetched_at"].dt.weekday

        tomorrow      = datetime.now(SGT) + timedelta(days=days_ahead)
        target_day    = tomorrow.weekday()
        overall_avg   = df["taxi_count"].mean()

        results = []
        for hour in self.PEAK_HOURS:
            # Historical avg for this hour + day of week
            mask     = (df["hour"] == hour) & (df["weekday"] == target_day)
            hour_df  = df[mask]

            if hour_df.empty:
                # Fall back to just hour average
                hour_df = df[df["hour"] == hour]

            if hour_df.empty:
                continue

            expected  = hour_df["taxi_count"].mean()
            vs_avg    = (expected - overall_avg) / overall_avg * 100

            # Rating based on vs_avg
            if expected >= overall_avg * 0.85:
                rating = "🟢 Good"
                advice = "Plenty of taxis expected"
            elif expected >= overall_avg * 0.65:
                rating = "🟡 Moderate"
                advice = "Some wait time expected"
            else:
                rating = "🔴 Poor"
                advice = "Plan ahead — taxis will be scarce!"

            results.append({
                "hour":       hour,
                "time_label": f"{hour:02d}:00",
                "day":        tomorrow.strftime("%A"),
                "expected":   round(expected, 1),
                "vs_avg_pct": round(vs_avg, 1),
                "rating":     rating,
                "advice":     advice,
            })

        return results


# ── 3. Day of Week Pattern ─────────────────────────────────────────────────────

class DayPatternAnalyser:
    """
    Computes historical average taxi count by hour × day of week.
    Used to generate a heatmap showing when taxis are most/least available.

    Output: DataFrame with columns [weekday, hour, avg_count, relative_pct]
    """

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    def __init__(self, district: str):
        self.district = district

    def get_pattern(self) -> pd.DataFrame:
        """
        Returns 7×24 heatmap of avg taxi availability.
        Perfect for Plotly imshow or heatmap chart.
        """
        rows = fetch_snapshots(self.district, minutes=7 * 24 * 60)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"])
        df["hour"]    = df["fetched_at"].dt.hour
        df["weekday"] = df["fetched_at"].dt.weekday
        df["day_name"] = df["weekday"].map(lambda x: self.DAY_NAMES[x])

        pattern = (
            df.groupby(["weekday", "day_name", "hour"])["taxi_count"]
            .mean()
            .reset_index()
            .rename(columns={"taxi_count": "avg_count"})
        )

        overall_avg = pattern["avg_count"].mean()
        pattern["relative_pct"] = (
            (pattern["avg_count"] - overall_avg) / overall_avg * 100
        ).round(1)

        # Classify each slot
        pattern["rating"] = pattern["relative_pct"].apply(
            lambda x: "🟢 High" if x > 10 else "🔴 Low" if x < -10 else "🟡 Normal"
        )

        return pattern.sort_values(["weekday", "hour"])

    def best_times(self, top_n: int = 5) -> list[dict]:
        """Return the N best times to find a taxi."""
        pattern = self.get_pattern()
        if pattern.empty:
            return []
        top = pattern.nlargest(top_n, "avg_count")
        return [
            {
                "day":       r["day_name"],
                "hour":      f"{int(r['hour']):02d}:00",
                "avg_count": round(r["avg_count"], 1),
                "rating":    r["rating"],
            }
            for _, r in top.iterrows()
        ]

    def worst_times(self, top_n: int = 5) -> list[dict]:
        """Return the N worst times to find a taxi."""
        pattern = self.get_pattern()
        if pattern.empty:
            return []
        bottom = pattern.nsmallest(top_n, "avg_count")
        return [
            {
                "day":       r["day_name"],
                "hour":      f"{int(r['hour']):02d}:00",
                "avg_count": round(r["avg_count"], 1),
                "rating":    r["rating"],
            }
            for _, r in bottom.iterrows()
        ]


# ── 4. HDB Price Forecast ──────────────────────────────────────────────────────

class HDBPriceForecaster:
    """
    Forecasts HDB resale prices for the next 6 months using
    linear trend extrapolation on monthly transaction data.

    Uses DuckDB historical data (9 years, 233k transactions).
    Falls back to simple linear regression if Prophet not available.
    """

    def __init__(self, town: str, flat_type: str = "4 ROOM"):
        self.town      = town.upper()
        self.flat_type = flat_type

    def _get_monthly_prices(self) -> pd.DataFrame:
        """Fetch monthly avg prices from DuckDB."""
        try:
            import duckdb
            con = duckdb.connect("data/hdb.duckdb", read_only=True)
            df  = con.execute(f"""
                SELECT
                    strptime(month, '%Y-%m')::date AS ds,
                    AVG(CAST(resale_price AS DOUBLE)) AS y
                FROM stg_hdb_raw
                WHERE town = '{self.town}'
                  AND flat_type = '{self.flat_type}'
                GROUP BY ds
                ORDER BY ds
            """).df()
            con.close()
            return df
        except Exception as e:
            log.warning("Could not fetch HDB prices: %s", e)
            return pd.DataFrame()

    def forecast(self, months_ahead: int = 6) -> pd.DataFrame:
        """
        Forecast next N months of HDB prices.
        Returns DataFrame: ds, yhat, yhat_lower, yhat_upper, is_forecast
        """
        df = self._get_monthly_prices()
        if df.empty or len(df) < 12:
            log.warning("[%s] Not enough price history", self.town)
            return pd.DataFrame()

        # Try Prophet first
        try:
            from prophet import Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            model.fit(df)
            future = model.make_future_dataframe(periods=months_ahead, freq="MS")
            forecast = model.predict(future)
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            result["is_forecast"] = result["ds"] > df["ds"].max()
            log.info("[%s] Prophet forecast complete: %d months ahead", self.town, months_ahead)
            return result

        except ImportError:
            log.info("Prophet not available — using linear trend fallback")
            return self._linear_forecast(df, months_ahead)

        except Exception as e:
            log.warning("Prophet failed: %s — using linear fallback", e)
            return self._linear_forecast(df, months_ahead)

    def _linear_forecast(self, df: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
        """Simple linear regression fallback when Prophet unavailable."""
        from sklearn.linear_model import LinearRegression

        df = df.copy()
        df["t"] = np.arange(len(df))

        X = df[["t"]].values
        y = df["y"].values

        model = LinearRegression()
        model.fit(X, y)

        # Historical
        df["yhat"]       = model.predict(X)
        df["yhat_lower"] = df["yhat"] * 0.95
        df["yhat_upper"] = df["yhat"] * 1.05
        df["is_forecast"] = False

        # Future
        future_rows = []
        last_date   = df["ds"].max()
        last_t      = df["t"].max()

        for i in range(1, months_ahead + 1):
            future_date = last_date + pd.DateOffset(months=i)
            t           = last_t + i
            yhat        = float(model.predict([[t]])[0])
            future_rows.append({
                "ds":          future_date,
                "yhat":        round(yhat, 0),
                "yhat_lower":  round(yhat * 0.95, 0),
                "yhat_upper":  round(yhat * 1.05, 0),
                "is_forecast": True,
            })

        result = pd.concat([
            df[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]],
            pd.DataFrame(future_rows)
        ], ignore_index=True)

        log.info("[%s] Linear forecast complete", self.town)
        return result

    def summary(self) -> dict | None:
        """Return a simple price forecast summary for the popup card."""
        forecast = self.forecast(months_ahead=6)
        if forecast.empty:
            return None

        historical = forecast[~forecast["is_forecast"]]
        future     = forecast[forecast["is_forecast"]]

        if historical.empty or future.empty:
            return None

        current_price = historical["yhat"].iloc[-1]
        future_price  = future["yhat"].iloc[-1]
        change        = future_price - current_price
        change_pct    = change / current_price * 100

        return {
            "current_price": round(current_price, 0),
            "forecast_price": round(future_price, 0),
            "change":         round(change, 0),
            "change_pct":     round(change_pct, 1),
            "trend":          "↑ Rising" if change > 0 else "↓ Falling",
            "months_ahead":   6,
        }
