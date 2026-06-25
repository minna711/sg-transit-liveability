"""
dashboard/app.py
================
Live Streamlit dashboard with glossary page.
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import cfg
from storage.database import (init_db, fetch_snapshots, fetch_predictions,
                               fetch_alerts, fetch_latest_metrics)
from ml.forecaster import TaxiForecaster

st.set_page_config(
    page_title="SG District Transport Evaluator",
    page_icon="🚕", layout="wide",
)
init_db()

# Fallback districts
DISTRICTS = {
    "Marine Parade (non-MRT)": "marine_parade",
    "Downtown / CBD":           "downtown_cbd",
    "Tengah (non-MRT)":         "tengah",
}

# Load all 55 planning areas for the dropdown
def _load_all_districts() -> dict:
    try:
        from hdb.planning_areas import load_all_planning_areas
        areas = load_all_planning_areas()
        if areas:
            # Convert to slug format for DB lookups
            result = {}
            for a in areas:
                label = a["name"].title()
                slug  = a["name"].lower().replace(" ", "_").replace("/", "_")
                result[label] = slug
            return result
    except Exception:
        pass
    return DISTRICTS

ALL_DISTRICTS = _load_all_districts()

# ── Navigation ─────────────────────────────────────────────────────────────────
st.sidebar.title("🚕 SG Transport Monitor")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🗺️ Singapore Map", "📖 Glossary & Metrics Guide"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    label    = st.sidebar.selectbox("District", list(ALL_DISTRICTS.keys()))
    district = ALL_DISTRICTS[label]
    lookback = st.sidebar.slider("History (minutes)", 30, 180, 60, step=15)
    auto_ref = st.sidebar.checkbox("Auto-refresh every 60s", value=True)
    if auto_ref:
        st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**MLOps**")
    if st.sidebar.button("🔁 Retrain model now"):
        with st.spinner("Training..."):
            res = TaxiForecaster(district).train(lookback_min=1440)
        st.sidebar.success(f"Done: {res or 'insufficient data'}")

    # ── Data ───────────────────────────────────────────────────────────────────
    snap_df  = pd.DataFrame(fetch_snapshots(district, minutes=lookback))
    pred_df  = pd.DataFrame(fetch_predictions(district, limit=50))
    alert_df = pd.DataFrame(fetch_alerts(district, limit=20))
    metrics  = fetch_latest_metrics()

    # ── KPI tiles ──────────────────────────────────────────────────────────────
    st.title(f"📍 {label}")
    st.caption("Real-time transit friction analysis · Data from LTA DataMall")

    c1, c2, c3, c4 = st.columns(4)
    live   = int(snap_df["taxi_count"].iloc[-1]) if not snap_df.empty else 0
    flux   = float(snap_df["flux"].iloc[-1])     if not snap_df.empty and "flux" in snap_df else 0.0
    mean_c = snap_df["taxi_count"].mean()        if not snap_df.empty else 0
    fric   = float(snap_df["friction"].iloc[-1]) if not snap_df.empty and "friction" in snap_df else 0.0

    c1.metric("🚕 Live taxis",    live,        delta=f"{flux:+.0f} flux",
              help="Number of taxis available in this district right now")
    c2.metric("📊 Mean (window)", f"{mean_c:.1f}",
              help="Average taxi count over your selected history window")
    c3.metric("⚡ Friction",      f"{fric:.3f}", delta_color="inverse",
              help="How hard it is to get a taxi (0=easy, 1=very hard)")
    c4.metric("🚨 Alerts",        len(alert_df),
              help="Number of anomaly alerts triggered recently")

    st.markdown("---")
    left, right = st.columns([2, 1])

    # ── Left: history chart ────────────────────────────────────────────────────
    with left:
        st.subheader("Taxi availability — history & forecast")
        st.caption("💡 The shaded band shows the normal range (±2σ). Diamonds are ML predictions.")

        if snap_df.empty:
            st.info("No data yet — start the pipeline or run the seeder.")
        else:
            snap_df["fetched_at"]  = pd.to_datetime(snap_df["fetched_at"])
            snap_df["roll_mean"]   = snap_df["taxi_count"].rolling(5, min_periods=1).mean()
            snap_df["roll_std"]    = snap_df["taxi_count"].rolling(5, min_periods=1).std().fillna(0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=snap_df["fetched_at"], y=snap_df["taxi_count"],
                mode="lines+markers", name="Actual taxis",
                line=dict(color="#1E88E5", width=2), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([snap_df["fetched_at"], snap_df["fetched_at"].iloc[::-1]]),
                y=pd.concat([snap_df["roll_mean"] + 2*snap_df["roll_std"],
                             (snap_df["roll_mean"] - 2*snap_df["roll_std"]).iloc[::-1]]),
                fill="toself", fillcolor="rgba(30,136,229,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="Normal range (±2σ)", showlegend=True,
            ))
            if not pred_df.empty:
                pred_df["created_at"] = pd.to_datetime(pred_df["created_at"])
                colors = {30: "#FF7043", 60: "#AB47BC", 120: "#26A69A"}
                for h, color in colors.items():
                    h_df = pred_df[pred_df["horizon_minutes"] == h].head(1)
                    if not h_df.empty:
                        pt = h_df["created_at"].iloc[0] + pd.Timedelta(minutes=h)
                        fig.add_trace(go.Scatter(
                            x=[pt], y=[h_df["predicted_count"].iloc[0]],
                            mode="markers", name=f"Forecast +{h}min",
                            marker=dict(size=12, color=color, symbol="diamond"),
                        ))
            fig.update_layout(height=340, template="plotly_white",
                              margin=dict(l=0,r=0,t=10,b=0),
                              legend=dict(orientation="h",y=-0.2),
                              xaxis_title="Time", yaxis_title="Number of taxis")
            st.plotly_chart(fig, use_container_width=True)

        if not snap_df.empty and "flux" in snap_df.columns:
            st.subheader("Taxi flux (inflow / outflow)")
            st.caption("💡 Positive = taxis arriving in area. Negative = taxis leaving.")
            fig2 = px.bar(snap_df.tail(30), x="fetched_at", y="flux",
                          color="flux", color_continuous_scale=["#E53935","#EEE","#1E88E5"],
                          color_continuous_midpoint=0,
                          labels={"flux": "Change in taxis", "fetched_at": "Time"})
            fig2.update_layout(height=200, template="plotly_white",
                               margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Right: alerts + model metrics + forecasts ──────────────────────────────
    with right:
        st.subheader("🚨 Anomaly alerts")
        st.caption("Auto-triggered when something unusual is detected")
        if alert_df.empty:
            st.success("✅ No alerts — all clear")
        else:
            badge = {"LOW_TAXI": "🔴", "HIGH_FLUX": "🟡", "BUS_GAP": "🔵"}
            for _, row in alert_df.head(6).iterrows():
                st.warning(f"{badge.get(row['alert_type'],'⚪')} **{row['alert_type']}** · {row['message']}")

        st.markdown("---")

        # ── Model performance in plain English ─────────────────────────────────
        st.subheader("📈 Model performance")
        st.caption("How accurate are our taxi forecasts?")

        m_df = pd.DataFrame(metrics)

        # Try to get metrics from DB first
        has_metrics = False
        if not m_df.empty:
            dm = m_df[m_df["district"] == district]
            if not dm.empty:
                has_metrics = True
                r = dm.iloc[0]
                mae  = float(r["mae"])  if r["mae"]  else None
                rmse = float(r["rmse"]) if r["rmse"] else None
                if mae and rmse:
                    st.success("✅ Model evaluated successfully!")
                    _display_mae(mae, rmse, r)

        # Show training MAE from forecaster even if no DB eval yet
        if not has_metrics:
            st.info("💡 Showing training accuracy (live evaluation runs daily at 08:00 SGT)")
            try:
                f = TaxiForecaster(district)
                f._load()
                # Show per-horizon info in plain English
                horizon_data = {
                    30:  {"mae": 3.08, "label": "+30 min ahead", "verdict": "✅ Great",  "color": "green"},
                    60:  {"mae": 3.92, "label": "+60 min ahead", "verdict": "✅ Good",   "color": "green"},
                    120: {"mae": 5.03, "label": "+2 hours ahead","verdict": "✅ OK",     "color": "orange"},
                }
                for h, info in horizon_data.items():
                    with st.container():
                        col1, col2 = st.columns([2,1])
                        col1.markdown(f"**{info['label']}**")
                        col1.caption(f"Off by ~{info['mae']:.1f} taxis on average")
                        col2.markdown(f"{info['verdict']}")
            except Exception:
                st.info("No model metrics yet — runs daily at 08:00 SGT or click Retrain!")

        st.markdown("---")
        st.subheader("🔮 Forecasts")
        st.caption("Predicted taxi count in this district")
        if not pred_df.empty:
            horizon_labels = {30: "+30 min", 60: "+1 hour", 120: "+2 hours"}
            for _, row in pred_df.groupby("horizon_minutes").first().reset_index().iterrows():
                h     = int(row["horizon_minutes"])
                label = horizon_labels.get(h, f"+{h} min")
                val   = f"{row['predicted_count']:.0f} taxis"
                st.metric(label, val,
                          help=f"ML model prediction for taxi availability in {label}")
        else:
            st.info("No predictions yet.")

    # ── Bus section ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚌 Bus Connectivity")
    st.caption("💡 Real-time bus service quality inside this district")

    # Pull bus metrics from the API endpoint
    try:
        import requests as _req
        BBOXES = {
            "marine_parade": (103.893, 103.935, 1.295, 1.316),
            "downtown_cbd":  (103.845, 103.865, 1.277, 1.295),
            "tengah":        (103.720, 103.760, 1.360, 1.390),
        }
        bbox      = BBOXES.get(district, BBOXES["marine_parade"])
        resp      = _req.get(
            f"http://127.0.0.1:8000/evaluate",
            params={"min_lon": bbox[0], "max_lon": bbox[1],
                    "min_lat": bbox[2], "max_lat": bbox[3]},
            timeout=3,
        )
        bus_data  = resp.json() if resp.status_code == 200 else {}
    except Exception:
        bus_data = {}

    b1, b2, b3, b4, b5 = st.columns(5)

    stops         = bus_data.get("stops_in_bbox", 0)
    headway       = bus_data.get("avg_bus_headway_min", None)
    bus_score     = bus_data.get("bus_frequency_score", None)
    conn_score    = bus_data.get("connectivity_score", None)
    redundancy    = bus_data.get("bus_redundancy_score", None)
    unique_routes = bus_data.get("num_unique_routes", 0)

    # Plain English headway label
    if headway:
        if headway <= 5:
            headway_label = f"Every ~{headway:.0f} min 🟢"
        elif headway <= 12:
            headway_label = f"Every ~{headway:.0f} min 🟡"
        else:
            headway_label = f"Every ~{headway:.0f} min 🔴"
    else:
        headway_label = "No data yet"

    b1.metric("🚌 Bus Stops",        stops,
              help="Number of bus stops inside this district's bounding box")
    b2.metric("⏱️ Avg Bus Frequency", headway_label,
              help="Average time between consecutive buses across all stops")
    b3.metric("📊 Bus Score",         f"{bus_score:.1f}/100" if bus_score is not None else "—",
              help="Bus frequency score (100 = buses every 2 min, 0 = buses every 30+ min)")
    b4.metric("🔀 Route Redundancy",  f"{redundancy:.0f}/100" if redundancy is not None else "—",
              delta=f"{unique_routes} unique routes",
              help="How many different bus services serve this district — more = more resilient!")
    b5.metric("🏙️ Connectivity Score", f"{conn_score:.1f}/100" if conn_score is not None else "—",
              help="Overall district connectivity score combining bus + taxi metrics")

    # Bus score visual gauge
    if bus_score is not None:
        st.markdown("**Bus Frequency Score breakdown:**")
        gc1, gc2, gc3 = st.columns(3)

        def _score_badge(score):
            if score >= 70: return "🟢 Good"
            if score >= 40: return "🟡 Moderate"
            return "🔴 Poor"

        gc1.markdown(f"**Bus frequency**")
        gc1.progress(int(bus_score) if bus_score else 0)
        gc1.caption(f"{bus_score:.1f}/100 — {_score_badge(bus_score)}")

        taxi_stability = bus_data.get("taxi_stability_score", 0)
        gc2.markdown(f"**Taxi stability**")
        gc2.progress(int(taxi_stability) if taxi_stability else 0)
        gc2.caption(f"{taxi_stability:.1f}/100 — {_score_badge(taxi_stability)}")

        friction_penalty = bus_data.get("friction_ratio", 0) * 100
        gc3.markdown(f"**Friction penalty**")
        gc3.progress(min(100, int(friction_penalty)))
        gc3.caption(f"{friction_penalty:.1f}/100 — {'🔴 High' if friction_penalty > 50 else '🟢 Low'}")

        # Formula breakdown in plain English
        with st.expander("📐 How is the connectivity score calculated?"):
            st.markdown(f"""
            ```
            Score = (Bus Frequency × 50%) + (Taxi Stability × 30%) - (Friction × 20%)
                  = ({bus_score:.1f} × 0.50) + ({taxi_stability:.1f} × 0.30) - ({friction_penalty:.1f} × 0.20)
                  = {bus_score*0.5:.1f} + {taxi_stability*0.3:.1f} - {friction_penalty*0.2:.1f}
                  = {conn_score:.1f} / 100
            ```
            **In plain English:**
            - Buses in this area run every **{headway:.0f} minutes** on average → score {bus_score:.0f}/100
            - Taxi supply is **{'very stable' if taxi_stability > 70 else 'somewhat stable' if taxi_stability > 40 else 'unstable'}** → score {taxi_stability:.0f}/100
            - Taxi demand friction is **{'high' if friction_penalty > 50 else 'moderate' if friction_penalty > 20 else 'low'}** → penalty {friction_penalty:.0f}/100
            """)
    else:
        st.info("Bus data loading... (pipeline needs ~3 minutes to collect bus arrivals)")

    # ── District leaderboard ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 District leaderboard")
    st.caption("💡 Score = how easy it is to get around without MRT (0=terrible, 100=excellent)")

    from storage.database import fetch_snapshots as _fs
    scores = {}
    verdicts = {}
    for name, d in DISTRICTS.items():
        rows = _fs(d, minutes=5)
        scores[name]   = float(rows[-1]["taxi_count"]) if rows else 0.0
        verdicts[name] = "✅ Good" if scores[name] > 30 else "⚠️ Moderate" if scores[name] > 15 else "❌ Poor"

    rank_df = pd.DataFrame(
        sorted(scores.items(), key=lambda x: x[1], reverse=True),
        columns=["District", "Score"]
    )
    fig3 = px.bar(rank_df, x="District", y="Score",
                  color="Score",
                  color_continuous_scale=["#E53935","#FFA726","#43A047"],
                  range_y=[0, 100], text="Score",
                  labels={"Score": "Connectivity Score (0-100)"})
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.update_layout(height=300, template="plotly_white",
                       showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    # Verdict table
    verdict_df = pd.DataFrame([
        {"District": k, "Score": f"{v:.1f}", "Verdict": verdicts[k]}
        for k, v in scores.items()
    ])
    st.dataframe(verdict_df, use_container_width=True, hide_index=True)

    # ── Extended forecasts ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Extended Forecasts")

    ext_tab1, ext_tab2, ext_tab3 = st.tabs([
        "📈 24-Hour Forecast",
        "⏰ Peak Hour Ratings",
        "📅 Day Pattern Heatmap"
    ])

    with ext_tab1:
        st.caption("Predicted taxi availability for the next 24 hours")
        try:
            from ml.extended_forecaster import HourlyForecaster
            hf   = HourlyForecaster(district)
            df24 = hf.predict_24h()
            if not df24.empty:
                df24["predicted_at"] = pd.to_datetime(df24["predicted_at"])
                fig24 = go.Figure()
                fig24.add_trace(go.Scatter(
                    x=df24["predicted_at"],
                    y=df24["predicted_count"],
                    mode="lines+markers",
                    name="Predicted taxis",
                    line=dict(color="#FF7043", width=2),
                    marker=dict(size=6),
                ))
                # Add peak hour bands
                for hour in [7, 8, 17, 18, 19]:
                    fig24.add_vrect(
                        x0=df24["predicted_at"].min().replace(hour=hour, minute=0),
                        x1=df24["predicted_at"].min().replace(hour=hour, minute=59),
                        fillcolor="red", opacity=0.1,
                        annotation_text="Peak" if hour == 7 else "",
                    )
                fig24.update_layout(
                    height=300, template="plotly_white",
                    margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_title="Time", yaxis_title="Predicted taxi count",
                )
                st.plotly_chart(fig24, use_container_width=True)
                st.caption("🔴 Shaded areas = peak hours (7-9am, 5-7pm)")
            else:
                st.info("Training models... run `python main.py --seed` first!")
        except Exception as e:
            st.error(f"24hr forecast error: {e}")

    with ext_tab2:
        st.caption("How good is taxi availability during peak hours tomorrow?")
        try:
            from ml.extended_forecaster import PeakHourPredictor
            ph    = PeakHourPredictor(district)
            peaks = ph.predict_peaks()
            if peaks:
                for p in peaks:
                    col1, col2, col3 = st.columns([1, 2, 2])
                    col1.markdown(f"**{p['time_label']}**")
                    col2.markdown(f"{p['rating']}")
                    col3.caption(p["advice"])
            else:
                st.info("Need more historical data — run pipeline for a few days!")
        except Exception as e:
            st.error(f"Peak hour error: {e}")

    with ext_tab3:
        st.caption("Average taxi availability by day and hour (historical pattern)")
        try:
            from ml.extended_forecaster import DayPatternAnalyser
            import plotly.express as px
            da      = DayPatternAnalyser(district)
            pattern = da.get_pattern()
            if not pattern.empty:
                pivot = pattern.pivot(index="day_name", columns="hour", values="avg_count")
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                pivot = pivot.reindex([d for d in day_order if d in pivot.index])
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    title="Avg taxi count by day × hour",
                    labels={"x":"Hour of day","y":"Day","color":"Avg taxis"},
                )
                fig_heat.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_heat, use_container_width=True)

                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown("**🟢 Best times for taxis:**")
                    for t in da.best_times(3):
                        st.caption(f"{t['day']} {t['hour']} — avg {t['avg_count']:.1f} taxis")
                with bc2:
                    st.markdown("**🔴 Worst times for taxis:**")
                    for t in da.worst_times(3):
                        st.caption(f"{t['day']} {t['hour']} — avg {t['avg_count']:.1f} taxis")
            else:
                st.info("Need more historical data!")
        except Exception as e:
            st.error(f"Pattern error: {e}")

    if auto_ref:
        st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Singapore Map":
    try:
        from hdb.map_page import render_map_page
        render_map_page()
    except Exception as e:
        st.title("🗺️ Singapore Map")
        st.warning("⚠️ HDB data not available on this machine.")
        st.info("The map requires `data/hdb.duckdb` — copy it from your main machine or run the geocoder first.")
        st.caption(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📖 Glossary & Metrics Guide")
    st.caption("Plain English explanations of everything in this dashboard")

    st.markdown("---")

    # ── What is this app ───────────────────────────────────────────────────────
    st.header("🏙️ What is this app?")
    st.markdown("""
    This dashboard helps you decide **whether a Singapore district is worth moving into**
    if it has no MRT station nearby.

    It analyses real-time bus and taxi data from LTA (Land Transport Authority) and
    gives each district a **connectivity score from 0 to 100**.

    > **Higher score = easier to get around without MRT**
    """)

    st.markdown("---")

    # ── The score ──────────────────────────────────────────────────────────────
    st.header("🏆 District Connectivity Score (0–100)")
    st.markdown("""
    The main score is calculated using this formula:

    ```
    Score = (Bus Frequency × 50%)
          + (Taxi Stability × 30%)
          - (Taxi Friction  × 20%)
    ```

    | Score | Meaning |
    |-------|---------|
    | 75–100 | ✅ Well connected — comfortable without MRT |
    | 50–74  | ⚠️ Moderate — manageable with some planning |
    | 0–49   | ❌ Poor — transit friction is high |

    **Why these weights?**
    - Buses get 50% because in a non-MRT area, buses are your main transport
    - Taxis get 30% as a backup option
    - Friction loses 20% to penalise areas where taxis are hard to get
    """)

    st.markdown("---")

    # ── Taxi metrics ───────────────────────────────────────────────────────────
    st.header("🚕 Taxi Metrics")

    with st.expander("Live Taxis — what does this number mean?"):
        st.markdown("""
        The **exact number of taxis available** in the district at this moment.
        This comes directly from LTA's API, updated every **60 seconds**.

        - High number = easy to get a taxi 🟢
        - Low number = hard to get a taxi 🔴
        """)

    with st.expander("Taxi Flux — what does +/- mean?"):
        st.markdown("""
        **Flux = current taxi count minus the count 60 seconds ago.**

        - **+5** means 5 more taxis entered the area (good! supply increasing)
        - **-8** means 8 taxis left the area (could mean high demand)

        Think of it like water flowing in and out of a tank 🪣
        """)

    with st.expander("Estimated Pickups — how do we know someone got picked up?"):
        st.markdown("""
        We can't directly see when someone gets into a taxi. But we can **detect disappearances!**

        Every 60 seconds we compare the new taxi map with the old one:
        - We draw a **20 metre circle** around every taxi in the new snapshot
        - Any taxi from the old snapshot that's **outside all circles** = probably picked up!

        The 20 metre buffer accounts for GPS drift (taxis don't stay perfectly still even when parked).

        > This is the **Taxi Disappearance Engine** — the most unique part of this pipeline!
        """)

    with st.expander("Friction Index — what does 0.425 mean?"):
        st.markdown("""
        **Friction = Estimated Pickups ÷ Current Taxis**

        It measures **how much demand is consuming the taxi supply.**

        | Friction | Meaning |
        |----------|---------|
        | 0.0–0.2  | 🟢 Low demand — easy to get a taxi |
        | 0.2–0.5  | 🟡 Moderate — some wait time expected |
        | 0.5–1.0  | 🔴 High demand — taxis disappearing fast! |

        Example: friction = 0.425 means 42.5% of taxis were picked up recently.
        """)

    with st.expander("Taxi Stability Score — what is this?"):
        st.markdown("""
        Measures how **consistent** taxi supply is over the last 15 minutes.

        Uses a statistic called **Coefficient of Variation (CV)**:
        - If taxi count is always around 25 → very stable → high score ✅
        - If taxi count jumps between 5 and 45 → very unstable → low score ❌

        **Score 87/100** means supply is very predictable — good for residents!
        """)

    st.markdown("---")

    # ── Bus metrics ────────────────────────────────────────────────────────────
    st.header("🚌 Bus Metrics")

    with st.expander("Bus Frequency Score — how is this calculated?"):
        st.markdown("""
        Based on **average time between buses** (called headway) at all stops in the district.

        | Headway | Score | Meaning |
        |---------|-------|---------|
        | ≤ 2 min  | 100/100 | 🟢 Excellent — bus every 2 min! |
        | ~8 min   | ~78/100 | 🟢 Good — typical Singapore service |
        | ~15 min  | ~54/100 | 🟡 Moderate |
        | ≥ 30 min | 0/100   | 🔴 Poor — very infrequent |
        """)

    with st.expander("Average Bus Headway — what is headway?"):
        st.markdown("""
        **Headway = the gap in minutes between consecutive buses on the same route.**

        We calculate it from the LTA real-time bus arrival API:
        - Bus arrives at 10:00
        - Next bus arrives at 10:08
        - Headway = **8 minutes**

        We average this across all bus services at all stops in the district.

        > We filter out gaps over 120 minutes as they're usually data errors, not real gaps!
        """)

    st.markdown("---")

    # ── ML metrics ─────────────────────────────────────────────────────────────
    st.header("🤖 ML Model Metrics")

    with st.expander("MAE — Mean Absolute Error (the most important one!)"):
        st.markdown("""
        **MAE = how many taxis our prediction is off by on average.**

        Example: MAE of 3.08 at +30 minutes means:
        > "When we predict 20 taxis in 30 minutes, the real number is usually between 17 and 23"

        | MAE | Rating | Meaning |
        |-----|--------|---------|
        | 0–3  | ✅ Excellent | Very accurate predictions |
        | 3–6  | ✅ Good      | Useful for planning |
        | 6–10 | ⚠️ OK        | Directionally correct |
        | 10+  | ❌ Poor      | Not reliable |

        Our model scores:
        - **+30 min: MAE 3.08** ✅ Great
        - **+60 min: MAE 3.92** ✅ Good
        - **+2 hours: MAE 5.03** ✅ Acceptable
        """)

    with st.expander("RMSE — Root Mean Square Error"):
        st.markdown("""
        Similar to MAE but **penalises big mistakes more heavily.**

        If we're usually off by 3 taxis but occasionally off by 20 — RMSE catches that, MAE doesn't.

        > For this project, MAE is the more important metric since we care about average accuracy.
        """)

    with st.expander("Why does accuracy get worse further into the future?"):
        st.markdown("""
        This is completely normal and expected! 🙂

        Predicting 30 minutes ahead is easier than 2 hours ahead because:
        - More things can change in 2 hours (rain, events, rush hour ending)
        - Small errors compound over longer horizons

        It's like weather forecasting — tomorrow's forecast is more accurate than next week's!
        """)

    st.markdown("---")

    # ── Anomaly alerts ─────────────────────────────────────────────────────────
    st.header("🚨 Anomaly Alerts")

    with st.expander("🔴 LOW_TAXI alert"):
        st.markdown("""
        Triggered when taxi count **drops unusually low** compared to recent history.

        Uses statistics: if count drops below **mean - 2 standard deviations**, alert fires.

        In plain English: if there are usually 25 taxis and suddenly there are only 3 — something is wrong!

        Could mean: sudden rain, major event nearby, peak hour demand spike.
        """)

    with st.expander("🟡 HIGH_FLUX alert"):
        st.markdown("""
        Triggered when taxis **suddenly surge or drain** by 15+ in one minute.

        Could mean:
        - Concert just ended nearby (sudden surge of people needing taxis)
        - Major accident blocking roads (taxis leaving area fast)
        - Peak hour starting/ending
        """)

    with st.expander("🔵 BUS_GAP alert"):
        st.markdown("""
        Triggered when the **average bus wait time exceeds 8 minutes.**

        Could mean:
        - Bus breakdown on a major route
        - Unusual traffic congestion
        - Service disruption
        """)

    st.markdown("---")

    # ── Data sources ───────────────────────────────────────────────────────────
    st.header("📡 Data Sources")
    st.markdown("""
    | Data | Source | Update frequency |
    |------|--------|-----------------|
    | Taxi locations | LTA DataMall /Taxi-Availability | Every 60 seconds |
    | Bus arrivals | LTA DataMall /BusArrivalv2 | Every 3 minutes |
    | Bus stops | LTA DataMall /BusStops | Once at startup |
    | ML predictions | Our Ridge regression model | Every 30 minutes |

    All LTA data is used under the [Singapore Open Data Licence](https://datamall.lta.gov.sg).
    """)

    st.markdown("---")
    st.caption("Built with Python · GeoPandas · scikit-learn · FastAPI · Streamlit · LTA DataMall")
