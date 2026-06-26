"""
ingestion/workers.py
====================
Their DataStore (RLock + capped list) and worker threading pattern,
extended to also write snapshots to SQLite for ML training.

Key upgrades vs original:
  - DataStore.push_taxi_snapshot() also calls storage.insert_snapshot()
  - BusWorker headway filter: gaps > 120 min are ignored (their fix)
  - _BaseWorker uses Event.wait() — stops instantly on stop()
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from config import cfg
from ingestion.client import get_bus_arrival, get_paginated

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


# ── Shared state ──────────────────────────────────────────────────────────────

class DataStore:
    """
    Thread-safe in-memory store shared between ingestion workers and the
    analytics/API layer.

    Uses threading.RLock (reentrant) so the same thread can safely call
    multiple DataStore methods without deadlocking.

    Taxi snapshots are capped at max_snapshots (default=60, ~1 hour at 60s
    intervals).  The ML layer reads from SQLite for longer history.
    """

    def __init__(self, max_snapshots: int = 60) -> None:
        self._lock = threading.RLock()
        self._max_snapshots = max_snapshots

        # Ordered list of (utc_timestamp, [raw LTA taxi dicts])
        self.taxi_snapshots: list[tuple[datetime, list[dict]]] = []

        # stop_code → list[service dicts] from BusArrivalv2
        self.bus_arrivals: dict[str, list[dict]] = {}

        # Full LTA bus-stop master list (fetched once at startup)
        self.bus_stops: list[dict] = []

        # Stop codes BusWorker actively refreshes
        self.monitored_stop_codes: set[str] = set()

    # ── Taxi ──────────────────────────────────────────────────────────────────

    def push_taxi_snapshot(self, records: list[dict]) -> None:
        """Append timestamped snapshot, evict oldest if at cap, persist to DB."""
        with self._lock:
            self.taxi_snapshots.append((datetime.now(tz=timezone.utc), records))
            if len(self.taxi_snapshots) > self._max_snapshots:
                self.taxi_snapshots.pop(0)

        # Persist to SQLite for ML training (outside lock — DB has its own safety)
        try:
            from storage.database import insert_snapshot
            from processing.spatial import filter_taxis_by_bbox
            from hdb.planning_areas import load_all_planning_areas

            # Load all 55 planning areas and save snapshot for each
            areas = load_all_planning_areas()
            if areas:
                for area in areas:
                    bbox  = (area["min_lon"], area["max_lon"],
                             area["min_lat"], area["max_lat"])
                    slug  = area["name"].lower().replace(" ", "_").replace("/","_").replace("-","_")
                    count = len(filter_taxis_by_bbox(records, bbox))
                    insert_snapshot(slug, count)
            else:
                # Fallback to 3 hardcoded districts
                FALLBACK = {
                    "marine_parade": (103.893, 103.935, 1.295, 1.316),
                    "downtown_cbd":  (103.845, 103.865, 1.277, 1.295),
                    "tengah":        (103.720, 103.760, 1.360, 1.390),
                }
                for district, bbox in FALLBACK.items():
                    count = len(filter_taxis_by_bbox(records, bbox))
                    insert_snapshot(district, count)
        except Exception as e:
            log.warning("Snapshot persist failed: %s", e)

    def get_last_two_snapshots(
        self,
    ) -> tuple[tuple[datetime, list[dict]] | None, tuple[datetime, list[dict]] | None]:
        """Return (T-1, T). Both None if fewer than 2 snapshots exist."""
        with self._lock:
            if len(self.taxi_snapshots) < 2:
                return None, None
            return self.taxi_snapshots[-2], self.taxi_snapshots[-1]

    def get_snapshots_within(self, minutes: int) -> list[tuple[datetime, list[dict]]]:
        """All snapshots whose timestamp falls within the last N minutes."""
        cutoff = datetime.now(tz=timezone.utc).timestamp() - minutes * 60
        with self._lock:
            return [
                (ts, recs)
                for ts, recs in self.taxi_snapshots
                if ts.timestamp() >= cutoff
            ]

    # ── Bus ───────────────────────────────────────────────────────────────────

    def push_bus_arrivals(self, stop_code: str, services: list[dict]) -> None:
        with self._lock:
            self.bus_arrivals[stop_code] = services

    def get_bus_arrivals(self, stop_codes: list[str]) -> dict[str, list[dict]]:
        with self._lock:
            return {sc: self.bus_arrivals.get(sc, []) for sc in stop_codes}

    def set_bus_stops(self, stops: list[dict]) -> None:
        with self._lock:
            self.bus_stops = stops

    def get_bus_stops(self) -> list[dict]:
        with self._lock:
            return list(self.bus_stops)

    def set_monitored_stops(self, codes: set[str]) -> None:
        with self._lock:
            self.monitored_stop_codes = codes

    def get_monitored_stops(self) -> set[str]:
        with self._lock:
            return set(self.monitored_stop_codes)


# ── Worker base ───────────────────────────────────────────────────────────────

class _BaseWorker(threading.Thread):
    """
    Daemon thread that calls _poll() on a fixed interval.
    Uses Event.wait() — stop() wakes the thread immediately.
    """

    interval: int = 60

    def __init__(self, store: DataStore) -> None:
        super().__init__(daemon=True, name=type(self).__name__)
        self._store      = store
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        log.info("%s started — polling every %d s", self.name, self.interval)
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception:
                log.exception("Unhandled error inside %s._poll()", self.name)
            # wait() returns early when stop() sets the event
            self._stop_event.wait(timeout=float(self.interval))
        log.info("%s stopped", self.name)

    def _poll(self) -> None:
        raise NotImplementedError


# ── Concrete workers ──────────────────────────────────────────────────────────

class TaxiWorker(_BaseWorker):
    """Fetches TaxiAvailability every 60 s and pushes to DataStore + SQLite."""

    interval = cfg.taxi_poll_interval

    def _poll(self) -> None:
        records = get_paginated("Taxi-Availability")
        if records:
            self._store.push_taxi_snapshot(records)
            log.debug("TaxiWorker: stored %d taxi coordinates", len(records))
        else:
            log.warning("TaxiWorker: empty payload — skipping snapshot")


class BusWorker(_BaseWorker):
    """
    Phase 1 (once): fetch full BusStops master list (~5 000 stops).
    Phase 2 (every 3 min): refresh live arrivals for monitored stops.
    """

    interval = cfg.bus_poll_interval

    def __init__(self, store: DataStore) -> None:
        super().__init__(store)
        self._stops_seeded = False

    def _poll(self) -> None:
        if not self._stops_seeded:
            log.info("BusWorker: fetching BusStops master list …")
            stops = get_paginated("BusStops")
            if stops:
                self._store.set_bus_stops(stops)
                self._stops_seeded = True
                log.info("BusWorker: cached %d bus stops", len(stops))
            else:
                log.warning("BusWorker: BusStops empty — retrying next cycle")
            return

        monitored = self._store.get_monitored_stops()
        if not monitored:
            log.debug("BusWorker: no monitored stops — waiting for bbox query")
            return

        for code in monitored:
            services = get_bus_arrival(code)
            self._store.push_bus_arrivals(code, services)
            time.sleep(0.05)   # 50 ms courtesy delay

        log.debug("BusWorker: refreshed %d stops", len(monitored))
