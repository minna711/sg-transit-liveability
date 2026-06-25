"""
analytics/engine.py
===================
Rolling-window metric computation and District Connectivity Score.
Based on Claude Code's engine.py with two fixes applied:
  1. Bus headway filter: gaps > 120 min discarded (their improvement)
  2. Taxi stability uses coefficient of variation (their improvement)

Score formula (weights from cfg):
  score = (bus_frequency_score × 0.50)
        + (taxi_stability_score × 0.30)
        − (friction_score × 0.20)   clamped [0, 100]
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

from config import cfg
from ingestion.workers import DataStore
from processing.spatial import filter_bus_stops_by_bbox, filter_taxis_by_bbox
from processing.taxi import detect_disappearances

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


@dataclass
class DistrictMetrics:
    bbox:                  BBox
    computed_at:           datetime
    taxi_count:            int
    taxi_flux:             int
    estimated_pickups:     int
    friction_ratio:        float
    taxi_stability_score:  float
    stops_in_bbox:         int
    avg_bus_headway_min:   float
    bus_frequency_score:   float
    connectivity_score:    float
    verdict:               str = ""


# ── Bus headway helpers ────────────────────────────────────────────────────────

def _parse_iso(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


def _headways_from_services(services: list[dict]) -> list[float]:
    """
    Extract inter-arrival gaps (minutes) from BusArrivalv2 service list.
    Filters out gaps > 120 min as data anomalies (their fix — prevents
    outliers from dragging the mean toward a falsely bad score).
    """
    now      = datetime.now(tz=timezone.utc)
    headways: list[float] = []

    for svc in services:
        arrivals: list[datetime] = []
        for key in ("NextBus", "NextBus2", "NextBus3"):
            bus_info = svc.get(key) or {}
            eta = _parse_iso(bus_info.get("EstimatedArrival", ""))
            if eta and eta > now:
                arrivals.append(eta)

        arrivals.sort()
        for i in range(1, len(arrivals)):
            gap_min = (arrivals[i] - arrivals[i - 1]).total_seconds() / 60.0
            if 0 < gap_min < 120:   # ← their fix: discard anomalous gaps
                headways.append(gap_min)

    return headways


# ── Score normalisation ────────────────────────────────────────────────────────

def _score_bus_frequency(avg_headway_min: float) -> float:
    """Linear interpolation: floor=2 min→100, ceiling=30 min→0."""
    lo, hi = cfg.bus_wait_floor_min, cfg.bus_wait_ceiling_min
    if avg_headway_min <= lo: return 100.0
    if avg_headway_min >= hi: return 0.0
    return 100.0 * (hi - avg_headway_min) / (hi - lo)


def _score_taxi_stability(
    window_snapshots: list[tuple[datetime, list[dict]]],
    bbox: BBox,
) -> float:
    """
    Coefficient of variation (CV = σ/μ) of in-bbox taxi counts.
    CV≈0 (very stable) → 100; CV≥1 (very volatile) → 0.
    Falls back to 50 (neutral) when < 2 snapshots exist.
    """
    if len(window_snapshots) < 2:
        return 50.0
    counts = [len(filter_taxis_by_bbox(r, bbox)) for _, r in window_snapshots]
    mean   = statistics.mean(counts)
    if mean == 0:
        return 0.0
    cv = statistics.stdev(counts) / mean
    return max(0.0, min(100.0, 100.0 * (1.0 - cv)))


def _verdict(score: float) -> str:
    if score >= 75: return "✅ Well connected — comfortable without MRT"
    if score >= 50: return "⚠️  Moderate connectivity — manageable with planning"
    return "❌ Poor connectivity — transit friction is high"


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_metrics(store: DataStore, bbox: BBox) -> DistrictMetrics:
    """Compute all transport metrics for bbox from the current DataStore state."""
    now = datetime.now(tz=timezone.utc)

    snap_prev, snap_curr = store.get_last_two_snapshots()
    if snap_curr is None:
        log.warning("compute_metrics: no snapshots yet — returning zeroed metrics")
        return _zeroed(bbox, now)

    _, curr_records = snap_curr
    curr_in_bbox = filter_taxis_by_bbox(curr_records, bbox)
    taxi_count   = len(curr_in_bbox)

    if snap_prev is not None:
        _, prev_records = snap_prev
        taxi_flux = taxi_count - len(filter_taxis_by_bbox(prev_records, bbox))
    else:
        prev_records = []
        taxi_flux    = 0

    # Disappearance engine
    if snap_prev is not None:
        d = detect_disappearances(prev_records, curr_records, bbox)
        friction_ratio    = d.friction_ratio
        estimated_pickups = d.estimated_pickups
    else:
        friction_ratio = estimated_pickups = 0

    # Taxi stability over rolling window
    window_snaps         = store.get_snapshots_within(cfg.rolling_window_minutes)
    taxi_stability_score = _score_taxi_stability(window_snaps, bbox)

    # Bus stops + headways
    all_stops      = store.get_bus_stops()
    stops_gdf      = filter_bus_stops_by_bbox(all_stops, bbox)
    stops_in_bbox  = len(stops_gdf)
    stop_codes     = stops_gdf["BusStopCode"].tolist() if not stops_gdf.empty else []
    arrivals_map   = store.get_bus_arrivals(stop_codes)

    all_headways: list[float] = []
    for services in arrivals_map.values():
        all_headways.extend(_headways_from_services(services))

    avg_bus_headway_min  = statistics.mean(all_headways) if all_headways else cfg.bus_wait_ceiling_min
    bus_frequency_score  = _score_bus_frequency(avg_bus_headway_min)

    # Composite score
    friction_100       = min(100.0, friction_ratio * 100.0)
    raw                = (
        bus_frequency_score   * cfg.bus_freq_weight
        + taxi_stability_score * cfg.taxi_stability_weight
        - friction_100        * cfg.taxi_friction_weight
    )
    connectivity_score = max(0.0, min(100.0, raw))

    log.info(
        "bbox %s → taxis=%d flux=%+d pickups=%d bus=%.1f taxi=%.1f fric=%.3f score=%.1f",
        bbox, taxi_count, taxi_flux, estimated_pickups,
        bus_frequency_score, taxi_stability_score, friction_ratio, connectivity_score,
    )

    return DistrictMetrics(
        bbox=bbox, computed_at=now,
        taxi_count=taxi_count, taxi_flux=taxi_flux,
        estimated_pickups=estimated_pickups, friction_ratio=friction_ratio,
        taxi_stability_score=taxi_stability_score,
        stops_in_bbox=stops_in_bbox, avg_bus_headway_min=avg_bus_headway_min,
        bus_frequency_score=bus_frequency_score,
        connectivity_score=connectivity_score,
        bus_redundancy_score=bus_redundancy_score,
        num_unique_routes=num_unique_routes,
        verdict=_verdict(connectivity_score),
    )


def _zeroed(bbox: BBox, ts: datetime) -> DistrictMetrics:
    return DistrictMetrics(
        bbox=bbox, computed_at=ts,
        taxi_count=0, taxi_flux=0, estimated_pickups=0,
        friction_ratio=0.0, taxi_stability_score=0.0,
        stops_in_bbox=0, avg_bus_headway_min=cfg.bus_wait_ceiling_min,
        bus_frequency_score=0.0, connectivity_score=0.0,
        verdict=_verdict(0.0),
    )
