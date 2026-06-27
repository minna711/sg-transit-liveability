"""
ml/anomaly.py
=============
Anomaly Detection Engine.
Thresholds are pulled from cfg so they're tunable without touching code.

Alert types:
  LOW_TAXI  — count falls below rolling mean - N*sigma
  HIGH_FLUX — absolute flux exceeds threshold
  BUS_GAP   — mean bus interval too long
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

SGT = timezone(timedelta(hours=8))

import numpy as np

from config import cfg
from storage.database import insert_alert, fetch_snapshots

log = logging.getLogger(__name__)


@dataclass
class Alert:
    district:     str
    alert_type:   str
    value:        float
    threshold:    float
    message:      str
    triggered_at: str = field(default_factory=lambda: datetime.now(SGT).isoformat())

    def to_dict(self) -> dict:
        return self.__dict__


class AnomalyDetector:
    def check(self, district: str, current_count: int,
              flux: float, bus_interval_s: float | None = None) -> list[Alert]:
        alerts  = []
        history = [r["taxi_count"] for r in fetch_snapshots(district, minutes=60)]

        # LOW_TAXI
        if len(history) >= 10:
            mean, std = np.mean(history), np.std(history)
            lower     = mean - cfg.anomaly_sigma * std
            if current_count < lower and current_count < mean * 0.5:
                msg = (f"{district}: only {current_count} taxis "
                       f"(mean={mean:.1f}, threshold={lower:.1f})")
                alerts.append(Alert(district, "LOW_TAXI", float(current_count),
                                    round(lower, 2), msg))
                insert_alert(district, "LOW_TAXI", float(current_count), round(lower, 2), msg)
                log.warning("🔴 LOW_TAXI: %s", msg)

        # HIGH_FLUX
        if abs(flux) >= cfg.anomaly_flux_thresh:
            direction = "surge" if flux > 0 else "drain"
            msg = (f"{district}: taxi {direction} of {flux:+.0f} "
                   f"(threshold=±{cfg.anomaly_flux_thresh})")
            alerts.append(Alert(district, "HIGH_FLUX", float(flux),
                                float(cfg.anomaly_flux_thresh), msg))
            insert_alert(district, "HIGH_FLUX", float(flux),
                         float(cfg.anomaly_flux_thresh), msg)
            log.warning("🟡 HIGH_FLUX: %s", msg)

        # BUS_GAP
        if bus_interval_s and bus_interval_s > cfg.anomaly_bus_thresh_s:
            msg = (f"{district}: bus gap {bus_interval_s:.0f}s "
                   f"exceeds {cfg.anomaly_bus_thresh_s:.0f}s")
            alerts.append(Alert(district, "BUS_GAP", round(bus_interval_s, 1),
                                cfg.anomaly_bus_thresh_s, msg))
            insert_alert(district, "BUS_GAP", round(bus_interval_s, 1),
                         cfg.anomaly_bus_thresh_s, msg)
            log.warning("🔵 BUS_GAP: %s", msg)

        if not alerts:
            log.debug("[%s] No anomalies (count=%d, flux=%.1f)", district, current_count, flux)

        return alerts
