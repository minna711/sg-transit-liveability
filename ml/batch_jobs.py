"""
ml/batch_jobs.py
================
MLOps batch scheduler using APScheduler.
  08:00 SGT daily  — retrain all district models
  08:05 SGT daily  — evaluate predictions vs actuals
  every 5 min      — fresh predictions + anomaly checks
  every 60 min     — extended predictions
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ml.forecaster import TaxiForecaster
from ml.extended_forecaster import HourlyForecaster, PeakHourPredictor, DayPatternAnalyser, HDBPriceForecaster
from ml.anomaly import AnomalyDetector
from storage.database import fetch_snapshots

log = logging.getLogger(__name__)

SGT = timezone(timedelta(hours=8))

DISTRICTS = ["marine_parade", "downtown_cbd", "tengah"]  # backward compat


def get_districts() -> list[str]:
    """Load all district slugs from planning_areas table."""
    try:
        from hdb.planning_areas import load_all_planning_areas
        areas = load_all_planning_areas()
        if areas:
            return [
                a["name"].lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                for a in areas
            ]
    except Exception as e:
        log.warning("Could not load planning areas: %s", e)
    return DISTRICTS  # fallback to 3 core districts


_detector = AnomalyDetector()


def job_train_all():
    log.info("=== Batch: TRAIN ALL (%s) ===", datetime.now(SGT).isoformat())
    for district in get_districts():
        try:
            TaxiForecaster(district).train(lookback_min=1440)
        except Exception as exc:
            log.exception("[%s] Training failed: %s", district, exc)


def job_evaluate_all():
    log.info("=== Batch: EVALUATE ALL (%s) ===", datetime.now(SGT).isoformat())
    for district in get_districts():
        try:
            TaxiForecaster(district).evaluate()
        except Exception as exc:
            log.exception("[%s] Evaluation failed: %s", district, exc)


def job_predict_and_check():
    log.info("=== Batch: PREDICT + ANOMALY CHECK (%s) ===", datetime.now(SGT).isoformat())
    for district in get_districts():
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
    log.info("=== Batch: EXTENDED PREDICTIONS (%s) ===", datetime.now(SGT).isoformat())
    for district in get_districts():
        try:
            hf   = HourlyForecaster(district)
            df24 = hf.predict_24h()
            log.info("[%s] 24hr forecast: %d hours", district, len(df24))

            ph    = PeakHourPredictor(district)
            peaks = ph.predict_peaks()
            log.info("[%s] Peak predictions: %d slots", district, len(peaks))

        except Exception as exc:
            log.exception("[%s] Extended prediction failed: %s", district, exc)

    try:
        for town in ["MARINE PARADE", "TAMPINES", "PUNGGOL", "TENGAH", "WOODLANDS"]:
            for flat_type in ["4 ROOM", "5 ROOM"]:
                hpf     = HDBPriceForecaster(town, flat_type)
                summary = hpf.summary()
                if summary:
                    log.info("[%s %s] Price forecast: %s", town, flat_type, summary["trend"])
    except Exception as exc:
        log.exception("HDB price forecast failed: %s", exc)


def create_batch_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="Asia/Singapore")

    sched.add_job(job_train_all, CronTrigger(hour=8, minute=0),
                  id="job_train_all", misfire_grace_time=300, replace_existing=True)

    sched.add_job(job_evaluate_all, CronTrigger(hour=8, minute=5),
                  id="job_evaluate_all", misfire_grace_time=300, replace_existing=True)

    sched.add_job(job_predict_and_check, IntervalTrigger(minutes=5),
                  id="job_predict_and_check", replace_existing=True)

    sched.add_job(job_extended_predictions, IntervalTrigger(minutes=60),
                  id="job_extended_predictions", replace_existing=True)

    log.info("Batch scheduler configured: %d jobs", len(sched.get_jobs()))
    return sched
