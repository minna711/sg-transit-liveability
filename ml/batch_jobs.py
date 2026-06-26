"""
ml/batch_jobs.py
================
MLOps batch scheduler using APScheduler.
  08:00 SGT daily  — retrain all district models
  08:05 SGT daily  — evaluate predictions vs actuals
  every 30 min     — fresh predictions + anomaly checks
"""
from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ml.forecaster import TaxiForecaster
from ml.extended_forecaster import HourlyForecaster, PeakHourPredictor, DayPatternAnalyser, HDBPriceForecaster
from ml.anomaly import AnomalyDetector
from storage.database import fetch_snapshots

log = logging.getLogger(__name__)

DISTRICTS = {
    "marine_parade": (103.893, 103.935, 1.295, 1.316),
    "downtown_cbd":  (103.845, 103.865, 1.277, 1.295),
    "tengah":        (103.720, 103.760, 1.360, 1.390),
}

_detector = AnomalyDetector()


def job_train_all():
    log.info("=== Batch: TRAIN ALL (%s) ===", datetime.utcnow().isoformat())
    for district in DISTRICTS:
        try:
            TaxiForecaster(district).train(lookback_min=1440)
        except Exception as exc:
            log.exception("[%s] Training failed: %s", district, exc)


def job_evaluate_all():
    log.info("=== Batch: EVALUATE ALL (%s) ===", datetime.utcnow().isoformat())
    for district in DISTRICTS:
        try:
            TaxiForecaster(district).evaluate()
        except Exception as exc:
            log.exception("[%s] Evaluation failed: %s", district, exc)


def job_predict_and_check():
    log.info("=== Batch: PREDICT + ANOMALY CHECK (%s) ===", datetime.utcnow().isoformat())
    for district in DISTRICTS:
        try:
            preds  = TaxiForecaster(district).predict()
            log.info("[%s] Predictions: %s", district, preds)
            recent = fetch_snapshots(district, minutes=5)
            if recent:
                _detector.check(district, recent[-1]["taxi_count"], recent[-1]["flux"])
        except Exception as exc:
            log.exception("[%s] Predict/check failed: %s", district, exc)


def job_extended_predictions():
    """Every hour — run extended predictions for all districts."""
    log.info("=== Batch: EXTENDED PREDICTIONS (%s) ===", datetime.utcnow().isoformat())
    for district in DISTRICTS:
        try:
            # 24-hour hourly forecast
            hf   = HourlyForecaster(district)
            df24 = hf.predict_24h()
            log.info("[%s] 24hr forecast: %d hours", district, len(df24))

            # Peak hour prediction
            ph     = PeakHourPredictor(district)
            peaks  = ph.predict_peaks()
            log.info("[%s] Peak predictions: %d slots", district, len(peaks))

        except Exception as exc:
            log.exception("[%s] Extended prediction failed: %s", district, exc)

    # HDB price forecasts (once daily is enough)
    try:
        for town in ["MARINE PARADE", "TAMPINES", "PUNGGOL", "TENGAH", "WOODLANDS"]:
            for flat_type in ["4 ROOM", "5 ROOM"]:
                hpf = HDBPriceForecaster(town, flat_type)
                summary = hpf.summary()
                if summary:
                    log.info("[%s %s] Price forecast: %s", town, flat_type, summary["trend"])
    except Exception as exc:
        log.exception("HDB price forecast failed: %s", exc)


def create_batch_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="Asia/Singapore")
    sched.add_job(job_train_all,       CronTrigger(hour=8, minute=0),
                  id="train",   misfire_grace_time=300, replace_existing=True)
    sched.add_job(job_evaluate_all,    CronTrigger(hour=8, minute=5),
                  id="evaluate", misfire_grace_time=300, replace_existing=True)
    sched.add_job(job_predict_and_check, IntervalTrigger(minutes=30),
                  id="predict",  replace_existing=True)
    # Extended predictions every hour
    sched.add_job(job_extended_predictions, IntervalTrigger(minutes=60),
                  id="extended_predict", replace_existing=True)

    log.info("Batch scheduler configured: %d jobs", len(sched.get_jobs()))
    return sched