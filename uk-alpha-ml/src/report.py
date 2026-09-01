"""
report.py
---------
Generates a full set of visual diagnostics for the project write-up:
  1. Equity curve + drawdown
  2. Rolling Sharpe ratio
  3. Long leg vs short leg cumulative contribution
  4. Monthly return heatmap (calendar view)
  5. Feature correlation heatmap
  6. Walk-forward CV AUC by fold and model

Run after run_pipeline.py:
    python -m src.report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def plot_equity_and_drawdown(bt: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(bt["Date"], bt["equity_curve"], color="#1f4e79", linewidth=1.5)
    axes[0].set_title("Strategy Equity Curve (long-short, net of costs)")
    axes[0].set_ylabel("Growth of £1")
    axes[0].grid(alpha=0.3)

    running_max = bt["equity_curve"].cummax()
    drawdown = bt["equity_curve"] / running_max - 1
    axes[1].fill_between(bt["Date"], drawdown, 0, color="#c0392b", alpha=0.5)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_equity_curve_and_drawdown.png", dpi=150)
    plt.close(fig)


def plot_rolling_sharpe(bt: pd.DataFrame, window: int = 26, periods_per_year: float = 50.4):
    """window is in REBALANCE PERIODS, not trading days (e.g. 26 five-day periods ~ 6 months)."""
    roll_mean = bt["net_return"].rolling(window).mean()
    roll_std = bt["net_return"].rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt["Date"], roll_sharpe, color="#2e7d32")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-period window)")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_rolling_sharpe.png", dpi=150)
    plt.close(fig)


def plot_long_short_legs(bt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    long_curve = (1 + bt["long_leg_return"]).cumprod()
    short_curve = (1 + bt["short_leg_return"]).cumprod()
    ax.plot(bt["Date"], long_curve, label="Long leg (top decile)", color="#1565c0")
    ax.plot(bt["Date"], short_curve, label="Short leg (bottom decile, inverted P&L)", color="#c62828")
    ax.plot(bt["Date"], bt["equity_curve"], label="Combined long-short", color="black", linewidth=2)
    ax.set_title("Long Leg vs Short Leg Contribution")
    ax.set_ylabel("Growth of £1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_long_short_legs.png", dpi=150)
    plt.close(fig)


def plot_monthly_returns_heatmap(bt: pd.DataFrame):
    df = bt.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    monthly = df.groupby(["year", "month"])["net_return"].apply(
        lambda r: (1 + r).prod() - 1
    ).unstack("month")

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(monthly))))
    im = ax.imshow(monthly.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index)
    for i in range(monthly.shape[0]):
        for j in range(monthly.shape[1]):
            val = monthly.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Monthly Net Returns (%)")
    plt.colorbar(im, ax=ax, label="Monthly return")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_monthly_returns_heatmap.png", dpi=150)
    plt.close(fig)


def plot_feature_correlation(feat: pd.DataFrame):
    feature_cols = [c for c in feat.columns if c.startswith("feat_")]
    corr = feat[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols, fontsize=8)
    ax.set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_feature_correlation.png", dpi=150)
    plt.close(fig)


def plot_cv_results(cv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, g in cv.groupby("model_name"):
        ax.plot(g["fold"], g["auc"], marker="o", label=name)
    ax.axhline(0.5, color="grey", linestyle="--", label="random baseline")
    ax.set_xlabel("Walk-forward CV fold")
    ax.set_ylabel("AUC")
    ax.set_title("Out-of-sample AUC by Fold and Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_cv_auc_by_fold.png", dpi=150)
    plt.close(fig)


def main():
    bt = pd.read_csv(RESULTS_DIR / "backtest_returns.csv", parse_dates=["Date"])
    cv = pd.read_csv(RESULTS_DIR / "cv_results.csv")
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    with open(RESULTS_DIR / "performance_summary.json") as f:
        metrics = json.load(f)

    plot_equity_and_drawdown(bt)
    plot_rolling_sharpe(bt)
    plot_long_short_legs(bt)
    plot_monthly_returns_heatmap(bt)
    plot_feature_correlation(feat)
    plot_cv_results(cv)

    print(f"Saved 6 plots to {RESULTS_DIR}/")
    print("\nPerformance summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
