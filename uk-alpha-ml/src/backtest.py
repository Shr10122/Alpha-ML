"""
backtest.py
-----------
A lean, transparent, fully custom vectorized backtester for a long-short
cross-sectional equity strategy. No external backtesting library -- every
line here is auditable, which matters a lot when you're explaining your
methodology to someone reviewing your project.

Strategy logic:
  On each rebalance date, rank stocks by predicted score.
  Go long the top decile, short the bottom decile, equal-weighted within
  each leg, dollar-neutral (long leg weight sums to +1, short leg to -1).
  Hold for `holding_period` trading days, then rebalance.

Costs:
  A simple proportional transaction cost is applied to turnover (fraction
  of portfolio weight that changes at each rebalance).
"""

import numpy as np
import pandas as pd


def build_portfolio_weights(
    scored_df: pd.DataFrame,
    score_col: str = "pred_score",
    top_frac: float = 0.1,
    bottom_frac: float = 0.1,
) -> pd.DataFrame:
    """
    scored_df: long-format df with columns ['Date', 'ticker', score_col].
    Returns df with an added 'weight' column, dollar-neutral within each date.
    """
    out = []
    for date, g in scored_df.groupby("Date"):
        g = g.dropna(subset=[score_col]).copy()
        n = len(g)
        if n < 10:  # need enough names to form meaningful deciles
            g["weight"] = 0.0
            out.append(g)
            continue

        n_top = max(1, int(np.floor(n * top_frac)))
        n_bottom = max(1, int(np.floor(n * bottom_frac)))

        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        g["weight"] = 0.0
        g.loc[:n_top - 1, "weight"] = 1.0 / n_top          # long leg, sums to +1
        g.loc[len(g) - n_bottom:, "weight"] = -1.0 / n_bottom  # short leg, sums to -1
        out.append(g)

    return pd.concat(out, ignore_index=True)


def run_backtest(
    weighted_df: pd.DataFrame,
    forward_return_col: str = "target_fwd_return",
    holding_period: int = 5,
    cost_per_unit_turnover: float = 0.0015,  # ~15 bps round-trip, realistic for liquid UK large-caps
) -> pd.DataFrame:
    """
    Computes the strategy's realized return stream, one observation per
    rebalance date, net of transaction costs from turnover.

    NOTE: forward_return_col must be the SAME horizon as holding_period, i.e.
    if holding_period=5, the target should be a 5-day forward return -- this
    keeps the backtest internally consistent with how the model was trained.
    """
    weighted_df = weighted_df.sort_values(["Date", "ticker"]).copy()

    # Gross return per date = sum(weight_i * forward_return_i) across stocks held that day
    gross = (
        weighted_df.assign(contrib=weighted_df["weight"] * weighted_df[forward_return_col])
        .groupby("Date")["contrib"].sum()
        .rename("gross_return")
    )

    # Per-leg returns (useful for diagnosing whether the long or short side is
    # driving performance, and for the "long-only vs long-short" comparison plot).
    long_leg = weighted_df[weighted_df["weight"] > 0]
    short_leg = weighted_df[weighted_df["weight"] < 0]
    long_return = (
        long_leg.assign(contrib=long_leg["weight"] * long_leg[forward_return_col])
        .groupby("Date")["contrib"].sum().rename("long_leg_return")
    )
    short_return = (
        short_leg.assign(contrib=short_leg["weight"].abs() * short_leg[forward_return_col] * -1)
        .groupby("Date")["contrib"].sum().rename("short_leg_return")
    )

    # Turnover estimate: sum of |weight change| per ticker between consecutive
    # rebalance dates, normalized. We approximate using absolute weight level
    # each period (a common simplification when previous-period weights per
    # ticker aren't tracked explicitly) -- documented here as a known
    # simplification of the backtest.
    turnover = (
        weighted_df.groupby("Date")["weight"].apply(lambda w: w.abs().sum())
        .rename("turnover")
    )

    result = pd.concat([gross, turnover, long_return, short_return], axis=1).reset_index()
    result["cost"] = result["turnover"] * cost_per_unit_turnover
    result["net_return"] = result["gross_return"] - result["cost"]
    result = result.sort_values("Date").reset_index(drop=True)
    result["equity_curve"] = (1 + result["net_return"]).cumprod()
    return result


def performance_metrics(result: pd.DataFrame, periods_per_year: float, return_col: str = "net_return") -> dict:
    r = result[return_col].dropna()
    if len(r) == 0:
        return {}

    mean_r = r.mean()
    std_r = r.std()
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else np.nan

    downside = r[r < 0]
    sortino = (mean_r / downside.std()) * np.sqrt(periods_per_year) if len(downside) > 1 and downside.std() > 0 else np.nan

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    n_periods = len(r)
    years = n_periods / periods_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    hit_rate = (r > 0).mean()

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "CAGR": cagr,
        "Hit Rate": hit_rate,
        "Total Return": equity.iloc[-1] - 1,
        "N Periods": n_periods,
    }
