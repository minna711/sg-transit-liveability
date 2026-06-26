"""
hdb/onemap_services.py
======================
OneMap API wrapper for real-time location services.
Used by the popup card to show live transport context for any HDB block.

Services:
  - get_nearest_mrt()       → closest MRT stations + distance
  - get_nearest_bus_stops() → closest bus stops + distance
  - get_walking_time()      → walk duration between two points
  - get_pt_commute_time()   → public transport duration to a destination
  - reverse_geocode()       → coordinates → address details

All results cached in memory (TTL 5 min) to avoid hammering the API.
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Singapore bounding box
SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.48
SG_LNG_MIN, SG_LNG_MAX = 103.6, 104.1

ONEMAP_BASE  = "https://www.onemap.gov.sg/api"
CACHE_TTL_S  = 300   # 5 minutes

# Simple in-memory cache: key → (expiry_time, data)
_cache: dict[str, tuple[float, any]] = {}

# CBD coordinates (used as default commute destination)
CBD_LAT, CBD_LNG = 1.2789, 103.8536


def _get_token() -> str:
    return os.environ.get("ONEMAP_TOKEN", "")


def _headers() -> dict:
    token = _get_token()
    return {"Authorization": token} if token else {}


def _cached_get(url: str, params: dict) -> dict | list | None:
    """GET with 5-minute in-memory cache."""
    cache_key = url + str(sorted(params.items()))
    now = time.time()

    if cache_key in _cache:
        expiry, data = _cache[cache_key]
        if now < expiry:
            return data

    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _cache[cache_key] = (now + CACHE_TTL_S, data)
        return data
    except Exception as e:
        log.warning("OneMap API error: %s", e)
        return None


# ── Postal code search ───────────────────────────────────────────────────────

def postal_to_coordinates(postal_code: str) -> dict | None:
    """
    Convert Singapore postal code to coordinates + address.
    Returns: {lat, lng, address, block, road, postal_code}
    """
    data = _cached_get(
        f"{ONEMAP_BASE}/common/elastic/search",
        {
            "searchVal":      postal_code,
            "returnGeom":     "Y",
            "getAddrDetails": "Y",
        },
    )
    if not data:
        return None

    results = data.get("results", [])
    for r in results:
        lat = float(r.get("LATITUDE", 0))
        lng = float(r.get("LONGITUDE", 0))
        if SG_LAT_MIN <= lat <= SG_LAT_MAX and SG_LNG_MIN <= lng <= SG_LNG_MAX:
            return {
                "lat":         lat,
                "lng":         lng,
                "address":     r.get("ADDRESS", ""),
                "block":       r.get("BLK_NO", ""),
                "road":        r.get("ROAD_NAME", ""),
                "postal_code": r.get("POSTAL", postal_code),
                "building":    r.get("BUILDING", ""),
            }
    return None


def get_live_bus_arrivals(stop_code: str, lta_api_key: str) -> list[dict]:
    """
    Get live bus arrivals for a specific stop from LTA API.
    Returns list of services with arrival times.
    """
    try:
        import requests as _req
        resp = _req.get(
            "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival",
            headers={"AccountKey": lta_api_key, "accept": "application/json"},
            params={"BusStopCode": stop_code},
            timeout=10,
        ).json()

        services = resp.get("Services", [])
        results  = []

        for svc in services:
            def parse_bus(bus_info):
                if not bus_info:
                    return None
                eta = bus_info.get("EstimatedArrival", "")
                if not eta:
                    return None
                from datetime import datetime, timezone
                try:
                    arr_time = datetime.fromisoformat(eta)
                    now      = datetime.now(tz=timezone.utc)
                    mins     = max(0, int((arr_time - now).total_seconds() / 60))
                    load     = bus_info.get("Load", "")
                    load_label = {"SEA": "🟢 Seated", "SDA": "🟡 Standing", "LSD": "🔴 Crowded"}.get(load, load)
                    return {"mins": mins, "load": load_label}
                except Exception:
                    return None

            next1 = parse_bus(svc.get("NextBus"))
            next2 = parse_bus(svc.get("NextBus2"))

            if next1:
                results.append({
                    "service":  svc.get("ServiceNo", ""),
                    "next1_min": next1["mins"],
                    "next2_min": next2["mins"] if next2 else None,
                    "load":     next1["load"],
                })

        results.sort(key=lambda x: x["next1_min"])
        return results

    except Exception as e:
        log.warning("Bus arrival fetch failed: %s", e)
        return []


def get_area_transport_profile(
    postal_code: str,
    radius_m: int = 500,
    lta_api_key: str = "",
) -> dict:
    """
    Full transport profile for a postal code within a given radius.
    
    Returns:
        address info, nearby bus stops + live arrivals, 
        nearest MRT, taxi count estimate, CBD commute time
    """
    # 1. Geocode postal code
    location = postal_to_coordinates(postal_code)
    if not location:
        return {"error": f"Could not find postal code {postal_code}"}

    lat, lng = location["lat"], location["lng"]

    # 2. Nearest MRT
    mrt = get_nearest_mrt_summary(lat, lng)

    # 3. Nearby bus stops
    bus_stops = get_nearest_bus_stops(lat, lng, radius_m=radius_m)

    # 4. Live bus arrivals for each stop
    bus_arrivals = []
    for stop in bus_stops[:5]:   # limit to 5 closest stops
        arrivals = get_live_bus_arrivals(stop["stop_code"], lta_api_key) if lta_api_key else []
        bus_arrivals.append({
            "stop_code":   stop["stop_code"],
            "description": stop["description"],
            "distance_m":  stop["distance_m"],
            "services":    arrivals[:4],   # top 4 services
        })

    # 5. CBD commute
    commute = get_pt_commute_time(lat, lng)

    return {
        "location":    location,
        "nearest_mrt": mrt,
        "bus_stops":   bus_arrivals,
        "num_stops":   len(bus_stops),
        "cbd_commute": commute,
        "radius_m":    radius_m,
    }


# ── Nearest MRT ───────────────────────────────────────────────────────────────

def get_nearest_mrt(lat: float, lng: float,
                    radius_m: int = 2000) -> list[dict]:
    """
    Get nearest MRT/LRT stations within radius.

    Returns list of dicts:
        name, latitude, longitude, distance_m, exit_code
    Sorted by distance ascending.
    """
    data = _cached_get(
        f"{ONEMAP_BASE}/public/popapi/getNearestMrt",
        {"latitude": lat, "longitude": lng, "radius_in_meters": radius_m},
    )
    if not data:
        return []

    results = []
    # Handle both list and dict responses
    stations = data if isinstance(data, list) else data.get("results", [])

    for s in stations:
        try:
            results.append({
                "name":       s.get("STATION_NAME", s.get("name", "Unknown")),
                "latitude":   float(s.get("LATITUDE", s.get("latitude", 0))),
                "longitude":  float(s.get("LONGITUDE", s.get("longitude", 0))),
                "distance_m": float(s.get("DISTANCE", s.get("distance", 0))),
                "exit_code":  s.get("EXIT_CODE", ""),
            })
        except (ValueError, KeyError):
            continue

    results.sort(key=lambda x: x["distance_m"])
    log.debug("Found %d MRT stations within %dm of (%.4f, %.4f)",
              len(results), radius_m, lat, lng)
    return results


def get_nearest_mrt_summary(lat: float, lng: float) -> dict:
    """
    Convenience: return the single nearest MRT station with a
    human-readable distance label.

    Returns: {name, distance_m, distance_label, walking_min}
    """
    stations = get_nearest_mrt(lat, lng, radius_m=2000)
    if not stations:
        return {"name": "No MRT nearby", "distance_m": 9999,
                "distance_label": ">2 km", "walking_min": None}

    nearest   = stations[0]
    dist_m    = nearest["distance_m"]
    walk_min  = round(dist_m / 80)   # avg walking speed 80m/min

    if dist_m < 500:
        label = f"{dist_m:.0f}m"
    else:
        label = f"{dist_m/1000:.1f}km"

    return {
        "name":           nearest["name"],
        "distance_m":     dist_m,
        "distance_label": label,
        "walking_min":    walk_min,
    }


# ── Nearest Bus Stops ─────────────────────────────────────────────────────────

def get_nearest_bus_stops(lat: float, lng: float,
                           radius_m: int = 500) -> list[dict]:
    """
    Get nearest bus stops within radius (default 500m).

    Returns list of dicts:
        stop_code, road_name, description, distance_m
    Sorted by distance ascending.
    """
    data = _cached_get(
        f"{ONEMAP_BASE}/public/popapi/getNearestBusStop",
        {"latitude": lat, "longitude": lng, "radius_in_meters": radius_m},
    )
    if not data:
        return []

    stops   = data if isinstance(data, list) else data.get("results", [])
    results = []

    for s in stops:
        try:
            results.append({
                "stop_code":   s.get("BUS_STOP_N", s.get("stop_code", "")),
                "road_name":   s.get("ROAD_NAME",  s.get("road_name", "")),
                "description": s.get("BUS_ROOF_N", s.get("description", "")),
                "distance_m":  float(s.get("DISTANCE", s.get("distance", 0))),
            })
        except (ValueError, KeyError):
            continue

    results.sort(key=lambda x: x["distance_m"])
    return results


def get_nearest_bus_stop_summary(lat: float, lng: float) -> dict:
    """Convenience: return nearest bus stop with human-readable label."""
    stops = get_nearest_bus_stops(lat, lng, radius_m=500)
    if not stops:
        return {"description": "No bus stop nearby", "distance_m": 9999,
                "distance_label": ">500m", "walking_min": None}

    nearest  = stops[0]
    dist_m   = nearest["distance_m"]
    walk_min = round(dist_m / 80)

    return {
        "stop_code":      nearest["stop_code"],
        "description":    nearest["description"] or nearest["road_name"],
        "distance_m":     dist_m,
        "distance_label": f"{dist_m:.0f}m",
        "walking_min":    walk_min,
        "num_stops":      len(stops),
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def get_walking_time(start_lat: float, start_lng: float,
                     end_lat: float,   end_lng: float) -> dict | None:
    """
    Get walking route between two points.

    Returns: {total_time_s, total_time_min, total_distance_m}
    """
    now  = datetime.now()
    data = _cached_get(
        f"{ONEMAP_BASE}/public/routingsvc/route",
        {
            "start":       f"{start_lat},{start_lng}",
            "end":         f"{end_lat},{end_lng}",
            "routeType":   "walk",
            "date":        now.strftime("%m-%d-%Y"),
            "time":        now.strftime("%H:%M:%S"),
            "mode":        "TRANSIT",
        },
    )
    if not data:
        return None

    try:
        total_time_s = float(data.get("route_summary", {}).get("total_time", 0))
        total_dist_m = float(data.get("route_summary", {}).get("total_distance", 0))
        return {
            "total_time_s":   total_time_s,
            "total_time_min": round(total_time_s / 60),
            "total_distance_m": total_dist_m,
        }
    except Exception:
        return None


def get_pt_commute_time(start_lat: float, start_lng: float,
                         end_lat: float = CBD_LAT,
                         end_lng: float = CBD_LNG) -> dict | None:
    """
    Get public transport commute time between two points.
    Defaults to CBD as destination.

    Returns: {total_time_s, total_time_min, total_distance_m}
    """
    now  = datetime.now()
    data = _cached_get(
        f"{ONEMAP_BASE}/public/routingsvc/route",
        {
            "start":       f"{start_lat},{start_lng}",
            "end":         f"{end_lat},{end_lng}",
            "routeType":   "pt",
            "date":        now.strftime("%m-%d-%Y"),
            "time":        now.strftime("%H:%M:%S"),
            "mode":        "TRANSIT",
            "numItineraries": 1,
        },
    )
    if not data:
        return None

    try:
        # PT response has itineraries
        itineraries = data.get("plan", {}).get("itineraries", [])
        if not itineraries:
            return None
        best = itineraries[0]
        total_time_s = float(best.get("duration", 0))
        return {
            "total_time_s":   total_time_s,
            "total_time_min": round(total_time_s / 60),
            "num_transfers":  best.get("transfers", 0),
        }
    except Exception:
        return None


# ── Reverse Geocode ───────────────────────────────────────────────────────────

def reverse_geocode(lat: float, lng: float,
                    buffer_m: int = 50,
                    address_type: str = "HDB") -> dict | None:
    """
    Convert coordinates to address details.

    Returns: {block, road, building, postal_code, address}
    """
    data = _cached_get(
        f"{ONEMAP_BASE}/public/revgeocode",
        {
            "location":    f"{lat},{lng}",
            "buffer":      buffer_m,
            "addressType": address_type,
            "otherFeatures": "N",
        },
    )
    if not data:
        return None

    try:
        results = data.get("GeocodeInfo", [])
        if not results:
            return None
        top = results[0]
        block    = top.get("BLOCK", "")
        road     = top.get("ROAD", "")
        building = top.get("BUILDINGNAME", "")
        postal   = top.get("POSTALCODE", "")
        address  = f"Blk {block} {road}" if block else road
        return {
            "block":       block,
            "road":        road,
            "building":    building,
            "postal_code": postal,
            "address":     address,
        }
    except Exception:
        return None


# ── Full block profile ─────────────────────────────────────────────────────────

def get_block_transport_profile(lat: float, lng: float) -> dict:
    """
    Get complete real-time transport profile for an HDB block.
    Used by the popup card in the dashboard.

    Returns all transport context in one call.
    """
    mrt      = get_nearest_mrt_summary(lat, lng)
    bus      = get_nearest_bus_stop_summary(lat, lng)
    address  = reverse_geocode(lat, lng)
    commute  = get_pt_commute_time(lat, lng)

    return {
        "address":    address,
        "nearest_mrt": mrt,
        "nearest_bus": bus,
        "cbd_commute": commute,
    }
