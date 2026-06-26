"""
ingestion/client.py
===================
LTA DataMall HTTP client.
Handles pagination ($skip), 429 rate-limiting, and exponential back-off.
All config pulled from the frozen cfg singleton.
"""
from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import cfg

log = logging.getLogger(__name__)


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "AccountKey": cfg.lta_api_key,
        "accept":     "application/json",
    })
    retry_policy = Retry(
        total=cfg.max_retries,
        backoff_factor=cfg.retry_backoff_s,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    return session


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def _get(endpoint: str, params: dict | None = None) -> dict:
    url  = cfg.lta_base_url + "/" + endpoint.lstrip("/")
    sess = _get_session()
    try:
        resp = sess.get(url, params=params, timeout=cfg.request_timeout)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", cfg.rate_limit_backoff_s))
            log.warning("Rate-limited by LTA. Sleeping %.0fs …", wait)
            time.sleep(wait)
            resp = sess.get(url, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        log.error("Timeout hitting %s", url)
        return {}
    except requests.exceptions.RequestException as exc:
        log.error("Request error: %s", exc)
        return {}


def get_paginated(endpoint: str) -> list[dict]:
    """
    Paginate through all LTA records (max 500 per page via $skip offset).
    Returns the full flat list.
    """
    records, skip = [], 0
    while True:
        payload = _get(endpoint, params={"$skip": skip})
        batch   = payload.get("value", [])
        if not batch:
            break
        records.extend(batch)
        if len(batch) < 500:   # last page
            break
        skip += 500
        time.sleep(0.2)        # polite delay
    return records


def get_bus_arrival(stop_code: str) -> list[dict]:
    """Fetch v3/BusArrival for one stop. Returns list of service dicts."""
    payload = _get("v3/BusArrival", params={"BusStopCode": stop_code})
    return payload.get("Services", [])
