"""
api.py
======
FastAPI app. Exposes:
  GET /evaluate          — district connectivity score
  GET /rank              — leaderboard of all known districts
  GET /predictions/{d}   — latest ML forecasts for a district
  GET /alerts            — recent anomaly alerts
  GET /health            — liveness check
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from analytics.engine import DistrictMetrics, compute_metrics
from ingestion.workers import DataStore

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

# Fallback hardcoded districts (used when planning areas not yet fetched)
KNOWN_DISTRICTS = {
    "marine_parade": (103.893, 103.935, 1.295, 1.316),
    "downtown_cbd":  (103.845, 103.865, 1.277, 1.295),
    "tengah":        (103.720, 103.760, 1.360, 1.390),
}


def get_all_districts() -> dict[str, tuple]:
    """
    Return all available districts as {name: bbox}.
    Uses all 55 planning areas from OneMap if available,
    falls back to hardcoded 3 districts.
    """
    try:
        from hdb.planning_areas import load_all_planning_areas
        areas = load_all_planning_areas()
        if areas:
            return {
                a["name"].title(): (a["min_lon"], a["max_lon"], a["min_lat"], a["max_lat"])
                for a in areas
            }
    except Exception:
        pass
    return KNOWN_DISTRICTS


class ScoreResponse(BaseModel):
    bbox:                 tuple
    taxi_count:           int
    taxi_flux:            int
    estimated_pickups:    int
    friction_ratio:       float
    taxi_stability_score: float
    stops_in_bbox:        int
    avg_bus_headway_min:  float
    bus_frequency_score:  float
    connectivity_score:   float
    verdict:              str


class RankEntry(BaseModel):
    rank:     int
    district: str
    score:    float
    verdict:  str


def evaluate_district(bbox: BBox, store: DataStore) -> DistrictMetrics:
    """Python-callable entry point (used by demo + tests)."""
    # Register stops in the bbox so BusWorker knows to poll them
    from processing.spatial import filter_bus_stops_by_bbox
    stops = filter_bus_stops_by_bbox(store.get_bus_stops(), bbox)
    if not stops.empty:
        store.set_monitored_stops(set(stops["BusStopCode"].tolist()))
    return compute_metrics(store, bbox)


def rank_districts(store: DataStore) -> list[dict]:
    """
    Tier-1 bonus: score ALL Singapore planning areas and return sorted leaderboard.
    Uses all 55 OneMap planning areas if available, falls back to 3 hardcoded districts.
    """
    districts = get_all_districts()
    results   = []
    for name, bbox in districts.items():
        m = compute_metrics(store, bbox)
        results.append({
            "district": name,
            "score":    m.connectivity_score,
            "verdict":  m.verdict,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def create_app(store: DataStore) -> FastAPI:
    app = FastAPI(
        title="SG District Transport Evaluator",
        description="Real-time transit friction scoring for Singapore districts.",
        version="2.0.0",
    )

    @app.get("/evaluate", response_model=ScoreResponse)
    def api_evaluate(min_lon: float, max_lon: float,
                     min_lat: float, max_lat: float):
        """
        Evaluate a district bounding box.
        Example (Marine Parade):
          GET /evaluate?min_lon=103.893&max_lon=103.935&min_lat=1.295&max_lat=1.316
        """
        bbox    = (min_lon, max_lon, min_lat, max_lat)
        metrics = evaluate_district(bbox, store)
        return ScoreResponse(
            bbox=bbox,
            taxi_count=metrics.taxi_count,
            taxi_flux=metrics.taxi_flux,
            estimated_pickups=metrics.estimated_pickups,
            friction_ratio=metrics.friction_ratio,
            taxi_stability_score=metrics.taxi_stability_score,
            stops_in_bbox=metrics.stops_in_bbox,
            avg_bus_headway_min=metrics.avg_bus_headway_min,
            bus_frequency_score=metrics.bus_frequency_score,
            connectivity_score=metrics.connectivity_score,
            verdict=metrics.verdict,
        )

    @app.get("/rank", response_model=list[RankEntry])
    def api_rank():
        """Return connectivity leaderboard for all known districts."""
        return [RankEntry(**r) for r in rank_districts(store)]

    @app.get("/predictions/{district}")
    def api_predictions(district: str, limit: int = 10):
        from storage.database import fetch_predictions
        return fetch_predictions(district, limit=limit)

    @app.get("/alerts")
    def api_alerts(district: Optional[str] = None, limit: int = 20):
        from storage.database import fetch_alerts
        return fetch_alerts(district, limit=limit)

    @app.get("/health")
    def health():
        return {"status": "ok", "snapshots": len(store.taxi_snapshots)}

    return app