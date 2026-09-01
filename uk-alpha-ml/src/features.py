"""
features.py
------------
Builds predictive features from raw OHLCV data, per-ticker, with strict
attention to look-ahead bias: every feature at row t uses only information
available up to and including t.

Also builds the forward-return TARGET, which is deliberately kept in a
separate, clearly-labelled column so it's never accidentally used as a feature.
"""

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram


def build_features_single_ticker(df: pd.DataFrame, forward_horizon: int = 5) -> pd.DataFrame:
    """
    df: single-ticker OHLCV dataframe, sorted by date ascending, with columns
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker'].
    forward_horizon: number of trading days ahead the target return looks.

    Returns a new dataframe with feature columns (prefixed 'feat_') and a
    target column 'target_fwd_return' / 'target_class'.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    close = df["Close"]
    volume = df["Volume"]

    # ---- Momentum / trend features (use only past data via pct_change/rolling) ----
    for w in [5, 10, 20, 60]:
        df[f"feat_return_{w}d"] = close.pct_change(w)

    df["feat_ma_5_20_ratio"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    df["feat_ma_20_60_ratio"] = close.rolling(20).mean() / close.rolling(60).mean() - 1
    df["feat_rsi_14"] = _rsi(close, 14)
    df["feat_macd_hist"] = _macd(close)

    # ---- Volatility features ----
    daily_ret = close.pct_change()
    df["feat_vol_10d"] = daily_ret.rolling(10).std()
    df["feat_vol_20d"] = daily_ret.rolling(20).std()
    rolling_high = df["High"].rolling(14).max()
    rolling_low = df["Low"].rolling(14).min()
    df["feat_atr_proxy_14"] = (rolling_high - rolling_low) / close

    # ---- Mean-reversion feature ----
    roll_mean_20 = close.rolling(20).mean()
    roll_std_20 = close.rolling(20).std()
    df["feat_price_zscore_20"] = (close - roll_mean_20) / roll_std_20.replace(0, np.nan)

    # ---- Volume features ----
    df["feat_volume_zscore_20"] = (
        (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)
    )
    obv = (np.sign(daily_ret.fillna(0)) * volume).cumsum()
    df["feat_obv_change_10d"] = obv.pct_change(10)

    # ---- Target: forward return over `forward_horizon` trading days ----
    # shift(-horizon) looks INTO THE FUTURE relative to row t -- this is fine
    # for a TARGET (label) but must NEVER be done for a feature column.
    df["target_fwd_return"] = close.shift(-forward_horizon) / close - 1

    return df


def build_features_all(raw_long_df: pd.DataFrame, forward_horizon: int = 5) -> pd.DataFrame:
    """Apply build_features_single_ticker to every ticker in a long-format dataframe."""
    out = []
    for ticker, g in raw_long_df.groupby("ticker"):
        out.append(build_features_single_ticker(g, forward_horizon=forward_horizon))
    result = pd.concat(out, ignore_index=True)
    return result


def add_cross_sectional_rank(df: pd.DataFrame, feature_col: str, new_col: str) -> pd.DataFrame:
    """
    Adds a cross-sectional percentile rank of `feature_col` computed WITHIN each
    date, across all tickers. This is a classic "relative strength" style alpha
    feature: it matters less whether a stock's momentum is positive in absolute
    terms and more whether it's high *relative to its peers on that day*.
    """
    df = df.copy()
    df[new_col] = df.groupby("Date")[feature_col].rank(pct=True)
    return df


def make_classification_target(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Converts the continuous 'target_fwd_return' into a cross-sectional quantile
    label per date (0 = worst performers, n_quantiles-1 = best performers).
    This is usually more robust to train on than raw returns.
    """
    df = df.copy()

    def _qcut_safe(x):
        try:
            return pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    df["target_class"] = df.groupby("Date")["target_fwd_return"].transform(_qcut_safe)
    return df
