import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Digital Asset Health Twin", layout="wide")

# ---------------------------------------------------------------
# Data loading (cached so it doesn't re-read on every click)
# ---------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@st.cache_data
def load_data():
    risk = pd.read_csv(os.path.join(DATA_DIR, "fused_risk_scores.csv"))
    ts = pd.read_csv(os.path.join(DATA_DIR, "sensor_timeseries.csv"))
    return risk, ts

risk_df, ts_df = load_data()

st.title("Digital Asset Health Twin")
st.caption("Physics-informed risk scoring for corroded pipeline segments — Track 4, Platinum Jubilee Innovation Hackathon")

tab1, tab2 = st.tabs(["Dashboard", "Methodology & Validation"])

# =================================================================
# TAB 1 — DASHBOARD
# =================================================================
with tab1:
    # ---- headline metrics ----
    total_assets = len(risk_df)
    high_risk = (risk_df["risk_score"] >= 50).sum()
    top20_n = int(total_assets * 0.20)
    top20 = risk_df.sort_values("risk_score", ascending=False).head(top20_n)
    capture_rate = top20["is_anomalous_label"].sum() / risk_df["is_anomalous_label"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assets Monitored", total_assets)
    c2.metric("High-Risk Assets (score ≥ 50)", int(high_risk))
    c3.metric("True Anomalies in Fleet", int(risk_df["is_anomalous_label"].sum()))
    c4.metric("Top-20% Capture Rate", f"{capture_rate:.0%}",
              help="Of all true anomalies, what fraction fall in the top 20% highest-risk assets")

    st.divider()

    # ---- sidebar filters ----
    with st.sidebar:
        st.header("Filters")
        materials = st.multiselect("Material", options=sorted(risk_df["material"].unique()),
                                    default=sorted(risk_df["material"].unique()))
        severities = st.multiselect("B31G Severity Class", options=sorted(risk_df["b31g_severity_class"].unique()),
                                     default=sorted(risk_df["b31g_severity_class"].unique()))
        min_score = st.slider("Minimum Risk Score", 0, 100, 0)

    filtered = risk_df[
        risk_df["material"].isin(materials) &
        risk_df["b31g_severity_class"].isin(severities) &
        (risk_df["risk_score"] >= min_score)
    ].sort_values("risk_score", ascending=False)

    st.subheader(f"Asset Risk Leaderboard ({len(filtered)} assets)")
    st.dataframe(
        filtered[["asset_id", "risk_score", "material", "service_type", "b31g_severity_class",
                  "wall_loss_pct", "rf_severity_proba", "is_anomalous_label"]],
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%.1f"
            ),
            "rf_severity_proba": st.column_config.NumberColumn("ML Severity Prob.", format="%.2f"),
            "wall_loss_pct": st.column_config.NumberColumn("Wall Loss %", format="%.1f%%"),
            "is_anomalous_label": st.column_config.CheckboxColumn("Confirmed Anomaly (synthetic ground truth)"),
        },
        hide_index=True,
        use_container_width=True,
        height=400,
    )

    st.divider()

    # ---- per-asset detail view ----
    st.subheader("Asset Detail")
    asset_id = st.selectbox("Select an asset", options=filtered["asset_id"].tolist())

    if asset_id:
        row = risk_df[risk_df.asset_id == asset_id].iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Risk Score", f"{row.risk_score:.1f} / 100")
        d2.metric("B31G Class", row.b31g_severity_class)
        d3.metric("Wall Loss", f"{row.wall_loss_pct:.1f}%")

        col_a, col_b = st.columns(2)

        with col_a:
            # risk score breakdown (the 50/35/15 fusion)
            def minmax(s, v):
                return (v - s.min()) / (s.max() - s.min() + 1e-9)
            components = pd.DataFrame({
                "Signal": ["ML Classifier (50%)", "B31G Formula (35%)", "Anomaly Detector (15%)"],
                "Contribution": [
                    50 * minmax(risk_df["rf_severity_proba"], row.rf_severity_proba),
                    35 * minmax(1 - risk_df["remaining_strength_factor"], 1 - row.remaining_strength_factor),
                    15 * minmax(risk_df["iso_max_anomaly_score"], row.iso_max_anomaly_score),
                ]
            })
            fig = px.bar(components, x="Contribution", y="Signal", orientation="h",
                         title="Risk Score Breakdown", range_x=[0, 55])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            asset_ts = ts_df[ts_df.asset_id == asset_id].sort_values("month_index")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=asset_ts.month_index, y=asset_ts.thickness_mm,
                                       mode="lines", name="Actual thickness"))
            fig2.update_layout(title="Wall Thickness Over Time (36 months)",
                                xaxis_title="Month", yaxis_title="Thickness (mm)")
            st.plotly_chart(fig2, use_container_width=True)

# =================================================================
# TAB 2 — METHODOLOGY & VALIDATION
# =================================================================
with tab2:
    st.subheader("How the risk score is built")
    st.markdown("""
    Each asset's 0–100 risk score fuses three independent signals so no single method's
    blind spot dominates the result:
    - **50% — ML Classifier**: Random Forest trained on engineered features (residual
      corrosion loss vs. a physics baseline, early-vs-late degradation acceleration, sensor stats)
    - **35% — ASME B31G**: an independent, non-ML industry-standard remaining-strength formula
    - **15% — Isolation Forest**: unsupervised anomaly detection, never trained on any ground-truth label
    """)

    st.subheader("Validation evidence")
    plot_dir = os.path.join(DATA_DIR, "plots")
    plots_to_show = [
        ("02_roc_curve.png", "Classifier ROC curve (test set)"),
        ("03_feature_importance.png", "What the classifier actually relies on"),
        ("12_learning_curve.png", "Bias/variance check — is it overfitting?"),
        ("14_applied_validation_water_leak.png",
         "Applied validation: our exact anomaly-detection method run on REAL independent leak-sensor data (AUC 0.93)"),
    ]
    for fname, caption in plots_to_show:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption, use_container_width=True)

    st.info(
        "Honesty note: the core dataset (150 assets) is synthetic, since real proprietary "
        "in-line inspection data isn't available. Performance numbers above are validated "
        "against two independent real-world datasets (water-leak sensors, pipe condition "
        "records) as an out-of-sample check, not presented as real-field results."
    )
