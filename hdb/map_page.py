"""
hdb/map_page.py
===============
Singapore interactive map page for Streamlit dashboard.
Shows:
  - HDB resale price heatmap (colour = price)
  - MRT stations overlaid
  - Taxi density heatmap
  - District bbox overlay
  - Click town → price trend chart + connectivity score
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

from hdb.analytics import (
    get_town_summary, get_block_prices,
    get_price_trend, get_available_towns, get_flat_types,
    get_value_for_money,
)
from storage.database import fetch_snapshots


# ── MRT station coordinates (key stations for overlay) ────────────────────────
MRT_STATIONS = [
    # EWL
    {"name": "Jurong East",    "lat": 1.3330, "lng": 103.7421, "line": "EWL"},
    {"name": "Buona Vista",    "lat": 1.3072, "lng": 103.7900, "line": "EWL"},
    {"name": "City Hall",      "lat": 1.2931, "lng": 103.8520, "line": "EWL"},
    {"name": "Tampines",       "lat": 1.3540, "lng": 103.9454, "line": "EWL"},
    {"name": "Pasir Ris",      "lat": 1.3731, "lng": 103.9494, "line": "EWL"},
    # NSL
    {"name": "Woodlands",      "lat": 1.4370, "lng": 103.7866, "line": "NSL"},
    {"name": "Yishun",         "lat": 1.4296, "lng": 103.8350, "line": "NSL"},
    {"name": "Ang Mo Kio",     "lat": 1.3699, "lng": 103.8495, "line": "NSL"},
    {"name": "Bishan",         "lat": 1.3510, "lng": 103.8480, "line": "NSL"},
    {"name": "Orchard",        "lat": 1.3047, "lng": 103.8318, "line": "NSL"},
    # CCL
    {"name": "Serangoon",      "lat": 1.3499, "lng": 103.8731, "line": "CCL"},
    {"name": "Dhoby Ghaut",    "lat": 1.2990, "lng": 103.8456, "line": "CCL"},
    {"name": "HarbourFront",   "lat": 1.2653, "lng": 103.8209, "line": "CCL"},
    # DTL
    {"name": "Bugis",          "lat": 1.3009, "lng": 103.8559, "line": "DTL"},
    {"name": "Little India",   "lat": 1.3066, "lng": 103.8493, "line": "DTL"},
    {"name": "Expo",           "lat": 1.3354, "lng": 103.9613, "line": "DTL"},
    # TEL
    {"name": "Marine Parade",  "lat": 1.3022, "lng": 103.9065, "line": "TEL"},
    {"name": "Bedok South",    "lat": 1.3215, "lng": 103.9401, "line": "TEL"},
]

MRT_LINE_COLORS = {
    "EWL": "#009645",   # green
    "NSL": "#D42E12",   # red
    "CCL": "#FA9E0D",   # orange
    "DTL": "#005EC4",   # blue
    "TEL": "#9D5B25",   # brown
}

# District bboxes for overlay
DISTRICT_BBOXES = {
    "MARINE PARADE": (103.893, 103.935, 1.295, 1.316),
    "CENTRAL AREA":  (103.845, 103.865, 1.277, 1.295),
    "TENGAH":        (103.720, 103.760, 1.360, 1.390),
}


def render_map_page():
    """Main function — call this from dashboard/app.py."""

    st.title("🗺️ Singapore HDB & Transport Map")
    st.caption("HDB resale prices overlaid with taxi density and MRT stations")

    # ── Sidebar controls ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Map controls")

    flat_types   = get_flat_types()
    default_idx  = flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0
    selected_flat = st.sidebar.selectbox("Flat type", flat_types, index=default_idx)
    months        = st.sidebar.slider("Months of data", 3, 24, 12, step=3)
    show_mrt      = st.sidebar.checkbox("Show MRT stations", value=True)
    show_taxi     = st.sidebar.checkbox("Show taxi heatmap", value=True)
    show_districts = st.sidebar.checkbox("Show district boxes", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("\u2696\ufe0f What matters to you?")
    st.sidebar.caption("Drag to set your priorities!")

    transport_w = st.sidebar.slider(
        "\U0001f68c Transport importance %", 0, 100, 50, step=5,
        help="How much do you care about buses and taxis?"
    )
    price_w = 100 - transport_w
    st.sidebar.caption(f"\U0001f3e0 Affordability: **{price_w}%**")

    if transport_w >= 70:
        st.sidebar.info("\U0001f68c Commuter — transport is everything!")
    elif transport_w <= 30:
        st.sidebar.info("\U0001f4b0 Budget buyer — price is king!")
    else:
        st.sidebar.info("\u2696\ufe0f Balanced buyer")

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading HDB data..."):
        try:
            town_df  = get_town_summary(flat_type=selected_flat, months=months)
            block_df = get_block_prices(flat_type=selected_flat, months=months)
            has_data = not town_df.empty
        except Exception as e:
            st.error(f"Could not load HDB data: {e}")
            st.info("Make sure geocoding has been run: `python hdb/geocoder.py`")
            return

    if not has_data:
        st.warning("No geocoded HDB data yet — run `python hdb/geocoder.py` first!")
        return

    # ── Main map ───────────────────────────────────────────────────────────────
    fig = go.Figure()

    # ── Layer 1: HDB price heatmap ─────────────────────────────────────────────
    if not block_df.empty:
        fig.add_trace(go.Densitymapbox(
            lat=block_df["latitude"],
            lon=block_df["longitude"],
            z=block_df["resale_price"],
            radius=25,
            colorscale="RdYlGn_r",   # red=expensive, green=affordable
            zmin=block_df["resale_price"].quantile(0.1),
            zmax=block_df["resale_price"].quantile(0.9),
            name="HDB Resale Price",
            colorbar=dict(
                title="Resale Price (S$)",
                x=1.02,
            ),
            opacity=0.7,
        ))

    # ── Layer 2: Town summary dots ─────────────────────────────────────────────
    if not town_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=town_df["lat"],
            lon=town_df["lng"],
            mode="markers+text",
            marker=dict(
                size=town_df["num_transactions"] / town_df["num_transactions"].max() * 20 + 8,
                color=town_df["avg_price"],
                colorscale="RdYlGn_r",
                showscale=False,
                opacity=0.85,
            ),
            text=town_df["town"].str.title(),
            textposition="top center",
            customdata=town_df[["avg_price", "num_transactions", "median_price"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg price: S$%{customdata[0]:,.0f}<br>"
                "Median: S$%{customdata[2]:,.0f}<br>"
                "Transactions: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            name="Towns",
        ))

    # ── Layer 3: MRT stations ──────────────────────────────────────────────────
    if show_mrt:
        mrt_df = pd.DataFrame(MRT_STATIONS)
        for line, color in MRT_LINE_COLORS.items():
            line_df = mrt_df[mrt_df["line"] == line]
            if line_df.empty:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=line_df["lat"],
                lon=line_df["lng"],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle"),
                text=line_df["name"],
                hovertemplate="<b>%{text}</b><br>" + line + " Line<extra></extra>",
                name=f"{line} Line",
                opacity=0.6,
            ))

    # ── Layer 4: District bbox overlays ───────────────────────────────────────
    if show_districts:
        for dname, (min_lon, max_lon, min_lat, max_lat) in DISTRICT_BBOXES.items():
            # Draw rectangle as a filled polygon
            fig.add_trace(go.Scattermapbox(
                lat=[min_lat, min_lat, max_lat, max_lat, min_lat],
                lon=[min_lon, max_lon, max_lon, min_lon, min_lon],
                mode="lines",
                line=dict(color="#00BCD4", width=2),
                fill="toself",
                fillcolor="rgba(0, 188, 212, 0.1)",
                name=dname,
                hoverinfo="name",
            ))

    # ── Layer 5: Taxi density ──────────────────────────────────────────────────
    if show_taxi:
        taxi_points = []
        for district_key in ["marine_parade", "downtown_cbd", "tengah"]:
            rows = fetch_snapshots(district_key, minutes=30)
            if rows:
                latest = rows[-1]
                # Use district center as proxy point weighted by count
                centers = {
                    "marine_parade": (1.3058, 103.9068),
                    "downtown_cbd":  (1.2850, 103.8550),
                    "tengah":        (1.3725, 103.7400),
                }
                if district_key in centers:
                    lat, lng = centers[district_key]
                    taxi_points.append({
                        "lat": lat, "lng": lng,
                        "count": latest["taxi_count"]
                    })
        if taxi_points:
            taxi_df = pd.DataFrame(taxi_points)
            fig.add_trace(go.Densitymapbox(
                lat=taxi_df["lat"],
                lon=taxi_df["lng"],
                z=taxi_df["count"],
                radius=40,
                colorscale=[[0,"rgba(0,0,255,0)"], [1,"rgba(0,0,255,0.5)"]],
                name="Taxi density",
                showscale=False,
                opacity=0.4,
            ))

    # ── Map layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=1.3521, lon=103.8198),
            zoom=11,
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="v",
            x=0, y=1,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Town price table ───────────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("💰 Price by town")
        st.caption(f"Average {selected_flat} resale price, last {months} months")

        display_df = town_df[["town", "avg_price", "median_price", "num_transactions"]].copy()
        display_df["town"]             = display_df["town"].str.title()
        display_df["avg_price"]        = display_df["avg_price"].apply(lambda x: f"S${x:,.0f}")
        display_df["median_price"]     = display_df["median_price"].apply(lambda x: f"S${x:,.0f}")
        display_df["num_transactions"] = display_df["num_transactions"].astype(int)
        display_df.columns             = ["Town", "Avg Price", "Median Price", "Transactions"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("🏆 Value-for-Money ranking")
        st.caption("Combines transport connectivity + affordability")

        # Get connectivity scores from API
        conn_scores = {}
        BBOXES = {
            "MARINE PARADE": (103.893, 103.935, 1.295, 1.316),
            "CENTRAL AREA":  (103.845, 103.865, 1.277, 1.295),
            "TENGAH":        (103.720, 103.760, 1.360, 1.390),
        }
        for town_name, bbox in BBOXES.items():
            try:
                r = requests.get("http://127.0.0.1:8000/evaluate",
                    params={"min_lon": bbox[0], "max_lon": bbox[1],
                            "min_lat": bbox[2], "max_lat": bbox[3]},
                    timeout=2).json()
                conn_scores[town_name.title()] = r.get("connectivity_score", 50.0)
            except Exception:
                conn_scores[town_name.title()] = 50.0

        vfm_df = get_value_for_money(town_df, conn_scores, transport_w/100, price_w/100)
        vfm_display = vfm_df[["town", "avg_price", "vfm_score", "vfm_verdict"]].head(10).copy()
        vfm_display["town"]      = vfm_display["town"].str.title()
        vfm_display["avg_price"] = vfm_display["avg_price"].apply(lambda x: f"S${x:,.0f}")
        vfm_display.columns      = ["Town", "Avg Price", "VFM Score", "Verdict"]
        st.dataframe(vfm_display, use_container_width=True, hide_index=True)

    # ── Price trend for selected town ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Price trend")

    towns        = get_available_towns()
    default_town = "MARINE PARADE" if "MARINE PARADE" in towns else towns[0]
    selected_town = st.selectbox(
        "Select town to see price trend",
        towns,
        index=towns.index(default_town) if default_town in towns else 0,
        format_func=lambda x: x.title(),
    )

    trend_df = get_price_trend(selected_town, flat_type=selected_flat)
    if not trend_df.empty:
        fig2 = px.line(
            trend_df, x="sale_month", y="avg_price",
            title=f"{selected_town.title()} — {selected_flat} monthly avg price",
            labels={"sale_month": "Month", "avg_price": "Average Price (S$)"},
        )
        fig2.update_traces(line=dict(color="#1E88E5", width=2))
        fig2.update_layout(
            height=300, template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_tickprefix="S$", yaxis_tickformat=",",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Key stats
        latest_price = trend_df["avg_price"].iloc[-1]
        oldest_price = trend_df["avg_price"].iloc[0]
        change       = latest_price - oldest_price
        change_pct   = change / oldest_price * 100

        s1, s2, s3 = st.columns(3)
        s1.metric("Latest avg price",  f"S${latest_price:,.0f}")
        s2.metric("Price change",       f"S${change:+,.0f}",
                  delta=f"{change_pct:+.1f}%")
        s3.metric("Total transactions", f"{trend_df['num_transactions'].sum():,}")
    else:
        st.info("No price trend data for this town yet.")

    # ── Block popup card ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📍 Block transport profile")
    st.caption("Enter any HDB block coordinates to see its real-time transport profile")

    col_lat, col_lng, col_btn = st.columns([2, 2, 1])
    input_lat = col_lat.number_input("Latitude",  value=1.3022, format="%.4f",
                                      help="e.g. 1.3022 for Marine Parade")
    input_lng = col_lng.number_input("Longitude", value=103.9068, format="%.4f",
                                      help="e.g. 103.9068 for Marine Parade")
    fetch_btn = col_btn.button("🔍 Fetch", use_container_width=True)

    if fetch_btn:
        with st.spinner("Fetching transport profile..."):
            try:
                from hdb.onemap_services import get_block_transport_profile
                profile = get_block_transport_profile(input_lat, input_lng)

                card_left, card_right = st.columns(2)

                with card_left:
                    addr = profile.get("address")
                    if addr:
                        st.markdown(f"### 📍 {addr.get('address', 'Unknown')}")
                        if addr.get("postal_code"):
                            st.caption(f"S{addr['postal_code']}")
                    else:
                        st.markdown(f"### 📍 ({input_lat:.4f}, {input_lng:.4f})")

                    st.markdown("---")

                    mrt      = profile.get("nearest_mrt", {})
                    mrt_name = mrt.get("name", "Unknown")
                    mrt_dist = mrt.get("distance_label", "?")
                    mrt_walk = mrt.get("walking_min", "?")
                    st.metric("🚇 Nearest MRT", mrt_name,
                              delta=f"{mrt_dist} · {mrt_walk} min walk" if mrt_walk else mrt_dist)

                    bus      = profile.get("nearest_bus", {})
                    bus_name = bus.get("description", "Unknown")
                    bus_dist = bus.get("distance_label", "?")
                    bus_walk = bus.get("walking_min", "?")
                    bus_num  = bus.get("num_stops", 0)
                    st.metric("🚌 Nearest bus stop", bus_name,
                              delta=f"{bus_dist} · {bus_walk} min walk · {bus_num} stops nearby")

                    commute = profile.get("cbd_commute")
                    if commute:
                        st.metric("⏱️ Commute to CBD",
                                  f"{commute['total_time_min']} min by PT",
                                  delta=f"{commute.get('num_transfers', 0)} transfer(s)")
                    else:
                        st.metric("⏱️ Commute to CBD", "N/A")

                with card_right:
                    try:
                        from hdb.analytics import get_block_prices
                        nearby = get_block_prices(flat_type=selected_flat, months=6)
                        if not nearby.empty:
                            nearby["dist"] = (
                                (nearby["latitude"]  - input_lat).abs() +
                                (nearby["longitude"] - input_lng).abs()
                            )
                            closest   = nearby.nsmallest(5, "dist")
                            avg_price = closest["resale_price"].mean()
                            min_price = closest["resale_price"].min()
                            max_price = closest["resale_price"].max()
                            st.metric("💰 Nearby avg price", f"S${avg_price:,.0f}",
                                      delta=f"S${min_price:,.0f} – S${max_price:,.0f} range")
                    except Exception:
                        st.metric("💰 Nearby avg price", "N/A")

                    try:
                        r = requests.get(
                            "http://127.0.0.1:8000/evaluate",
                            params={"min_lon": input_lng - 0.005,
                                    "max_lon": input_lng + 0.005,
                                    "min_lat": input_lat - 0.005,
                                    "max_lat": input_lat + 0.005},
                            timeout=3,
                        ).json()
                        score      = r.get("connectivity_score", 0)
                        verdict    = r.get("verdict", "")
                        taxi_count = r.get("taxi_count", 0)
                        bus_score  = r.get("bus_frequency_score", 0)
                        dist_m     = mrt.get("distance_m", 9999)

                        st.metric("📊 Connectivity score", f"{score:.1f}/100", delta=verdict)
                        st.metric("🚕 Live taxis nearby",  taxi_count)
                        st.metric("🚌 Bus frequency score", f"{bus_score:.1f}/100")

                        if   dist_m <= 500:  mrt_s, mrt_l = 100, "✅ Walking distance"
                        elif dist_m <= 1000: mrt_s, mrt_l =  70, "🟡 Short ride"
                        elif dist_m <= 2000: mrt_s, mrt_l =  40, "⚠️ Some distance"
                        else:                mrt_s, mrt_l =  10, "❌ Far from MRT"
                        st.metric("🚇 MRT proximity", f"{mrt_s}/100", delta=mrt_l)

                    except Exception:
                        st.info("Start the pipeline to see live scores!")

            except Exception as e:
                st.error(f"Error: {e}")