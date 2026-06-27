"""
main.py
=======
Orchestrator — their clean demo format + our ML/DB/scheduler layers.

Usage:
    python main.py --demo                         # offline mock demo
    python main.py --seed                         # seed 7 days of history
    LTA_API_KEY=<key> python main.py              # live pipeline + API
    streamlit run dashboard/app.py                # dashboard (separate terminal)
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timezone, timedelta

SGT = timezone(timedelta(hours=8)), timedelta, timezone

import uvicorn

from config import cfg
from storage.database import init_db, insert_snapshot
from ingestion.workers import DataStore, TaxiWorker, BusWorker
from analytics.engine import compute_metrics
from ml.batch_jobs import create_batch_scheduler, DISTRICTS
from ml.anomaly import AnomalyDetector
from ml.forecaster import TaxiForecaster
from api import create_app, evaluate_district, rank_districts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── District bboxes ────────────────────────────────────────────────────────────
BBOX_CBD          = (103.8198, 103.8698, 1.2600, 1.3000)
BBOX_MARINE_PARADE = (103.8900, 103.9250, 1.2900, 1.3150)
BBOX_TENGAH        = (103.7200, 103.7600, 1.3550, 1.3900)

DEMO_DISTRICTS = [
    ("Downtown CBD",            BBOX_CBD),
    ("Marine Parade (non-MRT)", BBOX_MARINE_PARADE),
    ("Tengah (new estate)",     BBOX_TENGAH),
]


# ── Mock store builder (their approach — real stop codes) ──────────────────────

def _build_mock_store() -> DataStore:
    store = DataStore()
    lon_lo, lon_hi = BBOX_MARINE_PARADE[0], BBOX_MARINE_PARADE[1]
    lat_lo, lat_hi = BBOX_MARINE_PARADE[2], BBOX_MARINE_PARADE[3]

    def _taxis(n):
        return [{"Latitude":  random.uniform(lat_lo, lat_hi),
                 "Longitude": random.uniform(lon_lo, lon_hi)} for _ in range(n)]

    # Three snapshots: shrinking supply simulates pickups
    store.push_taxi_snapshot(_taxis(30))
    store.push_taxi_snapshot(_taxis(27))
    store.push_taxi_snapshot(_taxis(24))

    # Real Marine Parade bus stop codes + ~8 min headways
    mock_stops = [
        {"BusStopCode": "92049", "Description": "Marine Parade Stn Exit A",
         "RoadName": "Marine Parade Rd", "Latitude": 1.3020, "Longitude": 103.9060},
        {"BusStopCode": "92051", "Description": "Roxy Sq",
         "RoadName": "East Coast Rd",    "Latitude": 1.3012, "Longitude": 103.9080},
        {"BusStopCode": "92059", "Description": "Katong Mall",
         "RoadName": "East Coast Rd",    "Latitude": 1.3008, "Longitude": 103.9100},
        {"BusStopCode": "92071", "Description": "Marine Parade RC",
         "RoadName": "Marine Parade Rd", "Latitude": 1.3035, "Longitude": 103.9040},
    ]
    store.set_bus_stops(mock_stops)
    store.set_monitored_stops({s["BusStopCode"] for s in mock_stops})

    now = datetime.now(tz=timezone.utc)
    def _services(routes, gap_min):
        return [{"ServiceNo": str(r), "Operator": "SBS",
                 "NextBus":  {"EstimatedArrival": (now + timedelta(minutes=gap_min*0.5)).isoformat()},
                 "NextBus2": {"EstimatedArrival": (now + timedelta(minutes=gap_min*1.5)).isoformat()},
                 "NextBus3": {"EstimatedArrival": (now + timedelta(minutes=gap_min*2.5)).isoformat()}}
                for r in routes]

    store.push_bus_arrivals("92049", _services([16, 31, 196], 8))
    store.push_bus_arrivals("92051", _services([16, 197, 43], 9))
    store.push_bus_arrivals("92059", _services([31, 36, 196], 7))
    store.push_bus_arrivals("92071", _services([196, 197],    10))
    return store


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    init_db()
    store    = _build_mock_store()
    detector = AnomalyDetector()
    W        = 64

    print()
    print("═" * W)
    print("  SG District Transport Connectivity  ·  MERGED DEMO")
    print("  (synthetic data — no LTA API key required)")
    print("═" * W)

    for name, bbox in DEMO_DISTRICTS:
        m      = evaluate_district(bbox, store)
        filled = int(m.connectivity_score / 5)
        bar    = "█" * filled + "░" * (20 - filled)

        print(f"\n  {'─' * (W - 4)}")
        print(f"  District : {name}")
        print(f"  Score    : {m.connectivity_score:5.1f}/100  [{bar}]")
        print(f"  Verdict  : {m.verdict}")
        print()
        print(f"  ┌─ Bus Frequency Score    {m.bus_frequency_score:6.1f} / 100")
        print(f"  ├─ Avg Bus Headway        {m.avg_bus_headway_min:6.1f} min")
        print(f"  ├─ Bus Stops in Bbox      {m.stops_in_bbox:6d}")
        print(f"  ├─ Taxi Stability Score   {m.taxi_stability_score:6.1f} / 100")
        print(f"  ├─ Taxi Count (current)   {m.taxi_count:6d}")
        print(f"  ├─ Taxi Flux (Δ vs T-1)  {m.taxi_flux:+6d}")
        print(f"  ├─ Estimated Pickups      {m.estimated_pickups:6d}")
        print(f"  └─ Friction Ratio         {m.friction_ratio:7.3f}")

        # Anomaly check
        alerts = detector.check(name.lower().replace(" ","_"),
                                m.taxi_count, float(m.taxi_flux))
        if alerts:
            for a in alerts:
                print(f"\n  🚨 ALERT [{a.alert_type}]: {a.message}")

    # Leaderboard
    print(f"\n  {'═' * (W - 4)}")
    print("  🏆 DISTRICT LEADERBOARD")
    print(f"  {'─' * (W - 4)}")
    for r in rank_districts(store):
        medals = {1:"🥇",2:"🥈",3:"🥉"}
        print(f"  {medals.get(r['rank'],'  ')} #{r['rank']}  "
              f"{r['district']:25s}  Score: {r['score']:5.1f}")

    # ML predictions
    print(f"\n  {'─' * (W - 4)}")
    print("  🤖 ML FORECASTS (EMA fallback — seed DB first for real model)")
    print(f"  {'─' * (W - 4)}")
    for district in DISTRICTS:
        preds = TaxiForecaster(district).predict()
        line  = "  ".join(f"+{h}min→{v:.1f}" for h, v in preds.items())
        print(f"  [{district}] {line}")

    print(f"\n  {'═' * (W - 4)}")
    print()
    print("  Next steps:")
    print("    python main.py --seed                ← seed 7 days of history")
    print("    streamlit run dashboard/app.py       ← open live dashboard")
    print("    LTA_API_KEY=<key> python main.py     ← live pipeline\n")


# ── Seed ───────────────────────────────────────────────────────────────────────

def run_seed() -> None:
    """Generate 7 days of synthetic history and train models."""
    init_db()
    print("\nSeeding 7 days of synthetic history...")
    base = {"marine_parade": 22, "downtown_cbd": 55, "tengah": 8}
    now  = datetime.now(SGT)
    total = 0
    for district, base_count in base.items():
        prev = base_count
        for mins_ago in range(7*24*60, 0, -1):
            ts   = now - timedelta(minutes=mins_ago)
            h    = ts.hour
            mult = (0.65 if 7<=h<10 else 0.60 if 17<=h<20
                    else 1.35 if 0<=h<5 else 0.80 if 12<=h<14 else 1.0)
            if ts.weekday() >= 5: mult *= 1.15
            rain  = -random.randint(5,12) if random.random() < 0.05 else 0
            count = max(0, int(base_count * mult + random.gauss(0,2) + rain))
            flux  = float(count - prev)
            insert_snapshot(district, count, flux=flux,
                            friction=round(min(1.0, abs(flux)/max(count,1)*0.3), 4))
            prev  = count
            total += 1

    print(f"Inserted {total:,} rows.")
    print("Training models...")
    for district in base:
        res = TaxiForecaster(district).train(lookback_min=7*24*60)
        print(f"  [{district}] {res or 'skipped (need more data)'}")
    print("✅ Seed complete!\n")


# ── Live server ────────────────────────────────────────────────────────────────

def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    if not cfg.lta_api_key:
        log.error("LTA_API_KEY is not set.")
        sys.exit(1)

    init_db()

    # Seed all 55 planning areas on startup
    try:
        from hdb.planning_areas import seed_planning_areas
        seed_planning_areas()
        log.info("Planning areas seeded ✅")
    except Exception as e:
        log.warning("Could not seed planning areas: %s", e)

    # Trigger initial prediction run after planning areas are seeded
    try:
        from ml.batch_jobs import job_predict_and_check
        log.info("Running initial predictions...")
        job_predict_and_check()
        log.info("Initial predictions done ✅")
    except Exception as e:
        log.warning("Initial prediction run failed: %s", e)

    store        = DataStore()
    batch_sched  = create_batch_scheduler()

    TaxiWorker(store).start()
    BusWorker(store).start()
    batch_sched.start()

    log.info("Workers started — warming up 10s …")
    time.sleep(10)

    app = create_app(store)
    log.info("API at http://%s:%d  |  Dashboard: streamlit run dashboard/app.py", host, port)

    # Run post-startup sanity check in background after API is ready
    import threading
    def _post_startup_check():
        time.sleep(5)  # wait for API to fully start
        try:
            from sanity_check import run_checks
            run_checks()
        except Exception as e:
            log.warning("Post-startup check failed: %s", e)
    threading.Thread(target=_post_startup_check, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SG District Transport Evaluator v2")
    parser.add_argument("--demo", action="store_true", help="Offline demo (no API key)")
    parser.add_argument("--seed", action="store_true", help="Seed 7 days of history")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.seed:
        run_seed()
    else:
        run_server(args.host, args.port)
