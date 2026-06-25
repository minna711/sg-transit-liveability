"""
hdb/map_page.py
===============
Singapore interactive map page for Streamlit dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

import hdb.analytics as hdb_analytics
from storage.database import fetch_snapshots

MRT_STATIONS = [
    # EWL (Green)
    {"name": "Jurong East",      "lat": 1.3330, "lng": 103.7421, "line": "EWL"},
    {"name": "Clementi",         "lat": 1.3152, "lng": 103.7651, "line": "EWL"},
    {"name": "Buona Vista",      "lat": 1.3072, "lng": 103.7900, "line": "EWL"},
    {"name": "Queenstown",       "lat": 1.2942, "lng": 103.8059, "line": "EWL"},
    {"name": "Outram Park",      "lat": 1.2800, "lng": 103.8395, "line": "EWL"},
    {"name": "City Hall",        "lat": 1.2931, "lng": 103.8520, "line": "EWL"},
    {"name": "Bugis",            "lat": 1.3009, "lng": 103.8559, "line": "EWL"},
    {"name": "Tampines",         "lat": 1.3540, "lng": 103.9454, "line": "EWL"},
    {"name": "Pasir Ris",        "lat": 1.3731, "lng": 103.9494, "line": "EWL"},
    # NSL (Red)
    {"name": "Jurong East",      "lat": 1.3330, "lng": 103.7421, "line": "NSL"},
    {"name": "Bukit Batok",      "lat": 1.3490, "lng": 103.7494, "line": "NSL"},
    {"name": "Choa Chu Kang",    "lat": 1.3853, "lng": 103.7443, "line": "NSL"},
    {"name": "Yew Tee",          "lat": 1.3969, "lng": 103.7474, "line": "NSL"},
    {"name": "Woodlands",        "lat": 1.4370, "lng": 103.7866, "line": "NSL"},
    {"name": "Yishun",           "lat": 1.4296, "lng": 103.8350, "line": "NSL"},
    {"name": "Ang Mo Kio",       "lat": 1.3699, "lng": 103.8495, "line": "NSL"},
    {"name": "Bishan",           "lat": 1.3510, "lng": 103.8480, "line": "NSL"},
    {"name": "Toa Payoh",        "lat": 1.3323, "lng": 103.8474, "line": "NSL"},
    {"name": "Novena",           "lat": 1.3204, "lng": 103.8438, "line": "NSL"},
    {"name": "Orchard",          "lat": 1.3047, "lng": 103.8318, "line": "NSL"},
    {"name": "Raffles Place",    "lat": 1.2830, "lng": 103.8513, "line": "NSL"},
    # NEL (Purple)
    {"name": "HarbourFront",     "lat": 1.2653, "lng": 103.8209, "line": "NEL"},
    {"name": "Outram Park",      "lat": 1.2800, "lng": 103.8395, "line": "NEL"},
    {"name": "Chinatown",        "lat": 1.2844, "lng": 103.8444, "line": "NEL"},
    {"name": "Little India",     "lat": 1.3066, "lng": 103.8493, "line": "NEL"},
    {"name": "Farrer Park",      "lat": 1.3124, "lng": 103.8545, "line": "NEL"},
    {"name": "Serangoon",        "lat": 1.3499, "lng": 103.8731, "line": "NEL"},
    {"name": "Hougang",          "lat": 1.3713, "lng": 103.8921, "line": "NEL"},
    {"name": "Punggol",          "lat": 1.4053, "lng": 103.9022, "line": "NEL"},
    # CCL (Orange)
    {"name": "Dhoby Ghaut",      "lat": 1.2990, "lng": 103.8456, "line": "CCL"},
    {"name": "Bishan",           "lat": 1.3510, "lng": 103.8480, "line": "CCL"},
    {"name": "Serangoon",        "lat": 1.3499, "lng": 103.8731, "line": "CCL"},
    {"name": "Paya Lebar",       "lat": 1.3180, "lng": 103.8922, "line": "CCL"},
    {"name": "one-north",        "lat": 1.2993, "lng": 103.7873, "line": "CCL"},
    {"name": "HarbourFront",     "lat": 1.2653, "lng": 103.8209, "line": "CCL"},
    # DTL (Blue)
    {"name": "Bukit Panjang",    "lat": 1.3784, "lng": 103.7761, "line": "DTL"},
    {"name": "Beauty World",     "lat": 1.3412, "lng": 103.7757, "line": "DTL"},
    {"name": "Botanic Gardens",  "lat": 1.3225, "lng": 103.8154, "line": "DTL"},
    {"name": "Stevens",          "lat": 1.3198, "lng": 103.8258, "line": "DTL"},
    {"name": "Newton",           "lat": 1.3124, "lng": 103.8380, "line": "DTL"},
    {"name": "Rochor",           "lat": 1.3041, "lng": 103.8521, "line": "DTL"},
    {"name": "Tampines",         "lat": 1.3540, "lng": 103.9454, "line": "DTL"},
    {"name": "Expo",             "lat": 1.3354, "lng": 103.9613, "line": "DTL"},
    # TEL (Brown)
    {"name": "Woodlands North",  "lat": 1.4480, "lng": 103.8002, "line": "TEL"},
    {"name": "Woodlands",        "lat": 1.4370, "lng": 103.7866, "line": "TEL"},
    {"name": "Springleaf",       "lat": 1.3993, "lng": 103.8192, "line": "TEL"},
    {"name": "Caldecott",        "lat": 1.3374, "lng": 103.8393, "line": "TEL"},
    {"name": "Stevens",          "lat": 1.3198, "lng": 103.8258, "line": "TEL"},
    {"name": "Orchard",          "lat": 1.3047, "lng": 103.8318, "line": "TEL"},
    {"name": "Great World",      "lat": 1.2937, "lng": 103.8227, "line": "TEL"},
    {"name": "Marine Parade",    "lat": 1.3022, "lng": 103.9065, "line": "TEL"},
    {"name": "Bedok South",      "lat": 1.3215, "lng": 103.9401, "line": "TEL"},
    {"name": "Sungei Bedok",     "lat": 1.3275, "lng": 103.9601, "line": "TEL"},
]

MRT_LINE_COLORS = {
    "EWL": "#009645",   # green
    "NSL": "#D42E12",   # red
    "NEL": "#9900AA",   # purple
    "CCL": "#FA9E0D",   # orange
    "DTL": "#005EC4",   # blue
    "TEL": "#9D5B25",   # brown
}

DISTRICT_BBOXES = {
    "MARINE PARADE": (103.893, 103.935, 1.295, 1.316),
    "CENTRAL AREA":  (103.845, 103.865, 1.277, 1.295),
    "TENGAH":        (103.720, 103.760, 1.360, 1.390),
}


def render_map_page():
    st.title("🗺️ Singapore HDB & Transport Map")
    st.caption("HDB resale prices overlaid with taxi density and MRT stations")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Map controls")

    flat_types    = hdb_analytics.get_flat_types()
    default_idx   = flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0
    selected_flat = st.sidebar.selectbox("Flat type", flat_types, index=default_idx)
    months        = st.sidebar.slider("Months of data", 3, 24, 12, step=3)
    show_mrt      = st.sidebar.checkbox("Show MRT stations", value=True)
    show_taxi     = st.sidebar.checkbox("Show taxi heatmap", value=True)
    show_districts = st.sidebar.checkbox("Show district boxes", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ What matters to you?")
    st.sidebar.caption("Drag to set your priorities!")
    transport_w = st.sidebar.slider("🚌 Transport importance %", 0, 100, 50, step=5)
    price_w     = 100 - transport_w
    st.sidebar.caption(f"🏠 Affordability: **{price_w}%**")
    if transport_w >= 70:
        st.sidebar.info("🚌 Commuter — transport is everything!")
    elif transport_w <= 30:
        st.sidebar.info("💰 Budget buyer — price is king!")
    else:
        st.sidebar.info("⚖️ Balanced buyer")

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading HDB data..."):
        try:
            town_df  = hdb_analytics.get_town_summary(flat_type=selected_flat, months=months)
            block_df = hdb_analytics.get_block_prices(flat_type=selected_flat, months=months)
            has_data = not town_df.empty
        except Exception as e:
            st.error(f"Could not load HDB data: {e}")
            st.info("Make sure geocoding has been run: `python hdb/geocoder.py`")
            return

    if not has_data:
        st.warning("No geocoded HDB data yet!")
        return

    # ── Leaflet Map ───────────────────────────────────────────────────────────
    from pathlib import Path
    import streamlit.components.v1 as components

    map_html_path = Path(__file__).parent.parent / "dashboard" / "sg_map.html"
    if map_html_path.exists():
        with open(map_html_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.caption("💡 Click anywhere on the map to see the connectivity score! Click an HDB area for full details.")
        components.html(map_html, height=540, scrolling=False)
    else:
        st.warning("Map file not found — make sure sg_map.html is in dashboard/ folder!")

    st.markdown("---")

    # ── Price table + VFM ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("💰 Price by town")
        st.caption(f"Average {selected_flat} resale price, last {months} months")
        display_df = town_df[["town","avg_price","median_price","num_transactions"]].copy()
        display_df["town"]             = display_df["town"].str.title()
        display_df["avg_price"]        = display_df["avg_price"].apply(lambda x: f"S${x:,.0f}")
        display_df["median_price"]     = display_df["median_price"].apply(lambda x: f"S${x:,.0f}")
        display_df["num_transactions"] = display_df["num_transactions"].astype(int)
        display_df.columns = ["Town","Avg Price","Median Price","Transactions"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("🏆 Value-for-Money ranking")
        st.caption("Combines transport connectivity + affordability")
        conn_scores = {}
        for town_name, bbox in DISTRICT_BBOXES.items():
            try:
                r = requests.get("http://127.0.0.1:8000/evaluate",
                    params={"min_lon":bbox[0],"max_lon":bbox[1],"min_lat":bbox[2],"max_lat":bbox[3]},
                    timeout=2).json()
                conn_scores[town_name.title()] = r.get("connectivity_score", 50.0)
            except Exception:
                conn_scores[town_name.title()] = 50.0

        vfm_df      = hdb_analytics.get_value_for_money(town_df, conn_scores, transport_w/100, price_w/100)
        vfm_display = vfm_df[["town","avg_price","vfm_score","vfm_verdict"]].head(10).copy()
        vfm_display["town"]      = vfm_display["town"].str.title()
        vfm_display["avg_price"] = vfm_display["avg_price"].apply(lambda x: f"S${x:,.0f}")
        vfm_display.columns = ["Town","Avg Price","VFM Score","Verdict"]
        st.dataframe(vfm_display, use_container_width=True, hide_index=True)

    # ── Price trend ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Price trend")
    towns        = hdb_analytics.get_available_towns()
    default_town = "MARINE PARADE" if "MARINE PARADE" in towns else towns[0]
    selected_town = st.selectbox(
        "Select town to see price trend", towns,
        index=towns.index(default_town) if default_town in towns else 0,
        format_func=lambda x: x.title(),
    )
    trend_df = hdb_analytics.get_price_trend(selected_town, flat_type=selected_flat)
    if not trend_df.empty:
        fig2 = px.line(trend_df, x="sale_month", y="avg_price",
                       title=f"{selected_town.title()} — {selected_flat} monthly avg price",
                       labels={"sale_month":"Month","avg_price":"Average Price (S$)"})
        fig2.update_traces(line=dict(color="#1E88E5", width=2))
        fig2.update_layout(height=300, template="plotly_white",
                           margin=dict(l=0,r=0,t=40,b=0),
                           yaxis_tickprefix="S$", yaxis_tickformat=",")
        st.plotly_chart(fig2, use_container_width=True)

        latest_price = trend_df["avg_price"].iloc[-1]
        oldest_price = trend_df["avg_price"].iloc[0]
        change       = latest_price - oldest_price
        change_pct   = change / oldest_price * 100
        s1, s2, s3   = st.columns(3)
        s1.metric("Latest avg price",   f"S${latest_price:,.0f}")
        s2.metric("Price change",        f"S${change:+,.0f}", delta=f"{change_pct:+.1f}%")
        s3.metric("Total transactions",  f"{trend_df['num_transactions'].sum():,}")
    else:
        st.info("No price trend data for this town yet.")

    # ── Block popup card ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📍 Block transport profile")
    st.caption("Enter a Singapore postal code to see real-time transport + HDB prices nearby")

    col_postal, col_radius, col_btn = st.columns([2, 2, 1])
    input_postal = col_postal.text_input("Postal code", value="440010",
                                          placeholder="e.g. 440010",
                                          help="6-digit Singapore postal code")
    radius_m     = col_radius.select_slider("Search radius",
                                            options=[250, 500, 1000],
                                            value=500,
                                            format_func=lambda x: f"{x}m")
    fetch_btn    = col_btn.button("🔍 Fetch", use_container_width=True)

    if fetch_btn:
        with st.spinner(f"Fetching transport profile for S{input_postal}..."):
            try:
                import os
                from hdb.onemap_services import get_area_transport_profile
                lta_key = os.environ.get("LTA_API_KEY", "")
                profile = get_area_transport_profile(input_postal, radius_m, lta_key)

                if "error" in profile:
                    st.error(profile["error"])
                else:
                    location = profile.get("location", {})
                    mrt      = profile.get("nearest_mrt", {})
                    commute  = profile.get("cbd_commute")
                    bus_stops = profile.get("bus_stops", [])

                    # ── Address header ─────────────────────────────────────────
                    st.markdown(f"### 📍 {location.get('address', input_postal)}")
                    st.caption(f"S{location.get('postal_code', input_postal)} · "
                               f"Showing within {radius_m}m radius")
                    st.markdown("---")

                    card_left, card_right = st.columns(2)

                    with card_left:
                        # MRT
                        st.metric("🚇 Nearest MRT",
                                  mrt.get("name", "Unknown"),
                                  delta=f"{mrt.get('distance_label','?')} · "
                                        f"{mrt.get('walking_min','?')} min walk")

                        # Commute
                        if commute:
                            st.metric("⏱️ Commute to CBD",
                                      f"{commute['total_time_min']} min by PT")
                        else:
                            st.metric("⏱️ Commute to CBD", "N/A")

                        # Nearby HDB prices
                        try:
                            lat = location.get("lat", 0)
                            lng = location.get("lng", 0)
                            nearby_df = hdb_analytics.get_block_prices(flat_type=selected_flat, months=6)
                            if not nearby_df.empty:
                                nearby_df["dist"] = (
                                    (nearby_df["latitude"]  - lat).abs() +
                                    (nearby_df["longitude"] - lng).abs()
                                )
                                closest   = nearby_df.nsmallest(5, "dist")
                                avg_price = closest["resale_price"].mean()
                                min_price = closest["resale_price"].min()
                                max_price = closest["resale_price"].max()
                                st.metric("💰 Nearby avg price", f"S${avg_price:,.0f}",
                                          delta=f"S${min_price:,.0f}–S${max_price:,.0f}")
                        except Exception:
                            st.metric("💰 Nearby avg price", "N/A")

                        # MRT proximity score
                        dist_m = mrt.get("distance_m", 9999)
                        if   dist_m <= 500:  mrt_s, mrt_l = 100, "✅ Walking distance"
                        elif dist_m <= 1000: mrt_s, mrt_l =  70, "🟡 Short ride"
                        elif dist_m <= 2000: mrt_s, mrt_l =  40, "⚠️ Some distance"
                        else:                mrt_s, mrt_l =  10, "❌ Far from MRT"
                        st.metric("🚇 MRT proximity score", f"{mrt_s}/100", delta=mrt_l)

                    with card_right:
                        # Live connectivity from API
                        try:
                            lat = location.get("lat", 0)
                            lng = location.get("lng", 0)
                            r = requests.get(
                                "http://127.0.0.1:8000/evaluate",
                                params={"min_lon": lng-0.005, "max_lon": lng+0.005,
                                        "min_lat": lat-0.005, "max_lat": lat+0.005},
                                timeout=3,
                            ).json()
                            st.metric("📊 Connectivity score",
                                      f"{r.get('connectivity_score',0):.1f}/100",
                                      delta=r.get("verdict",""))
                            st.metric("🚕 Live taxis nearby", r.get("taxi_count", 0))
                            st.metric("🚌 Bus frequency score",
                                      f"{r.get('bus_frequency_score',0):.1f}/100")
                            st.metric("🚌 Bus stops nearby", profile.get("num_stops", 0),
                                      delta=f"within {radius_m}m")
                        except Exception:
                            st.info("Start the pipeline for live scores!")

                    # ── Live bus arrivals ──────────────────────────────────────
                    if bus_stops:
                        st.markdown("---")
                        st.subheader(f"🚌 Live bus arrivals within {radius_m}m")
                        for stop in bus_stops:
                            if not stop["services"]:
                                continue
                            with st.expander(
                                f"🚏 {stop['description']} ({stop['stop_code']}) "
                                f"· {stop['distance_m']:.0f}m away"
                            ):
                                for svc in stop["services"]:
                                    cols = st.columns([1, 2, 2])
                                    cols[0].markdown(f"**{svc['service']}**")
                                    cols[1].markdown(f"🕐 {svc['next1_min']} min")
                                    if svc.get("next2_min"):
                                        cols[1].caption(f"then {svc['next2_min']} min")
                                    cols[2].caption(svc.get("load", ""))
                    else:
                        st.info(f"No bus stops found within {radius_m}m")

            except Exception as e:
                st.error(f"Error: {e}")
