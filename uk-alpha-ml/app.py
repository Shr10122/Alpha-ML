"""
app.py
------
Interactive Streamlit dashboard for the UK Alpha ML project. Reads the
artifacts produced by `python -m src.run_pipeline` (results/*.csv,
results/*.json, data/processed/features.parquet) and renders them as an
interactive web app -- this is also the app we containerize and deploy to AWS.

Run locally:
    streamlit run app.py

Requires run_pipeline.py to have been run at least once so results/ is populated.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="UK Alpha ML Dashboard", layout="wide")


@st.cache_data
def load_data():
    bt = pd.read_csv(RESULTS_DIR / "backtest_returns.csv", parse_dates=["Date"])
    cv = pd.read_csv(RESULTS_DIR / "cv_results.csv")
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(RESULTS_DIR / "performance_summary.json") as f:
        metrics = json.load(f)
    return bt, cv, feat, metrics


st.title("📈 Machine Learning for Alpha Generation & Backtesting")
st.caption("A cross-sectional long-short strategy on UK (FTSE 100) equities, with leakage-safe backtesting.")

try:
    bt, cv, feat, metrics = load_data()
except FileNotFoundError:
    st.error(
        "No results found. Run `python -m src.run_pipeline` first to generate "
        "the data this dashboard reads from."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
col2.metric("CAGR", f"{metrics['CAGR']*100:.1f}%")
col3.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.1f}%")
col4.metric("Hit Rate", f"{metrics['Hit Rate']*100:.1f}%")
col5.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Equity curve + drawdown
# ---------------------------------------------------------------------------
st.subheader("Equity Curve & Drawdown")

running_max = bt["equity_curve"].cummax()
drawdown = bt["equity_curve"] / running_max - 1

fig = go.Figure()
fig.add_trace(go.Scatter(x=bt["Date"], y=bt["equity_curve"], name="Equity Curve",
                         line=dict(color="#1f4e79", width=2)))
fig.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Growth of £1")
st.plotly_chart(fig, use_container_width=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=bt["Date"], y=drawdown, fill="tozeroy",
                            line=dict(color="#c0392b"), name="Drawdown"))
fig_dd.update_layout(height=220, margin=dict(t=20, b=20), yaxis_title="Drawdown")
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# Strategy Returns Breakdown (Safely handled)
# ---------------------------------------------------------------------------
st.subheader("Strategy Performance Breakdown")

fig_legs = go.Figure()

# Plot long/short legs if present in DataFrame
if "long_return" in bt.columns and "short_return" in bt.columns:
    long_curve = (1 + bt["long_return"]).cumprod()
    short_curve = (1 + bt["short_return"]).cumprod()
    fig_legs.add_trace(go.Scatter(x=bt["Date"], y=long_curve, name="Long leg", line=dict(color="#1565c0")))
    fig_legs.add_trace(go.Scatter(x=bt["Date"], y=short_curve, name="Short leg", line=dict(color="#c62828")))
elif "net_return" in bt.columns:
    net_curve = (1 + bt["net_return"]).cumprod()
    fig_legs.add_trace(go.Scatter(x=bt["Date"], y=net_curve, name="Cumulative Net Return", line=dict(color="#2e7d32")))

# Always plot combined equity curve
fig_legs.add_trace(go.Scatter(x=bt["Date"], y=bt["equity_curve"], name="Combined Equity",
                               line=dict(color="black", width=2, dash="dash")))
fig_legs.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Growth of £1")
st.plotly_chart(fig_legs, use_container_width=True)

# ---------------------------------------------------------------------------
# CV results
# ---------------------------------------------------------------------------
st.subheader("Walk-Forward Cross-Validation: Model Comparison")
col_a, col_b = st.columns([2, 1])

with col_a:
    fig_cv = px.line(cv, x="fold", y="auc", color="model_name", markers=True,
                     title="Out-of-sample AUC by fold")
    fig_cv.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="random baseline")
    fig_cv.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_cv, use_container_width=True)

with col_b:
    summary = cv.groupby("model_name")["auc"].agg(["mean", "std"]).reset_index()
    summary.columns = ["Model", "Mean AUC", "Std AUC"]
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        "AUCs around 0.50-0.53 are expected for daily equity return prediction "
        "on liquid large-caps -- a much higher score here would suggest leakage, "
        "not genuine skill."
    )

# ---------------------------------------------------------------------------
# Monthly returns heatmap
# ---------------------------------------------------------------------------
st.subheader("Monthly Returns Heatmap")
df = bt.copy()
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month_name().str[:3]
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly = df.groupby(["year", "month"])["net_return"].apply(lambda r: (1 + r).prod() - 1).reset_index()
pivot = monthly.pivot(index="year", columns="month", values="net_return").reindex(columns=month_order)

fig_hm = px.imshow(pivot * 100, color_continuous_scale="RdYlGn", zmin=-15, zmax=15,
                    text_auto=".1f", aspect="auto", labels=dict(color="Return %"))
fig_hm.update_layout(height=300, margin=dict(t=20, b=20))
st.plotly_chart(fig_hm, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature explorer
# ---------------------------------------------------------------------------
st.subheader("Feature Explorer")
feature_cols = [c for c in feat.columns if c.startswith("feat_")]
tickers = sorted(feat["ticker"].unique())

col_x, col_y, col_t = st.columns(3)
with col_x:
    chosen_feature = st.selectbox("Feature to inspect", feature_cols)
with col_y:
    chosen_ticker = st.selectbox("Ticker", ["All tickers"] + tickers)
with col_t:
    st.write("")

plot_df = feat if chosen_ticker == "All tickers" else feat[feat["ticker"] == chosen_ticker]
fig_feat = px.line(plot_df, x="Date", y=chosen_feature, color="ticker" if chosen_ticker == "All tickers" else None,
                    title=f"{chosen_feature} over time")
fig_feat.update_layout(height=350, margin=dict(t=40, b=20), showlegend=(chosen_ticker == "All tickers"))
st.plotly_chart(fig_feat, use_container_width=True)

st.divider()
st.caption(
    "Research/educational project — not investment advice. See README.md for "
    "full methodology and known limitations (survivorship bias, simplified "
    "transaction costs, no slippage model)."
)