"""
FOR TESTING/VALIDATION ONLY -- not part of the real project deliverable.

Generates fake-but-realistic multi-ticker OHLCV data so the pipeline
(features -> CV -> models -> backtest) can be validated end-to-end without
needing live internet access. When you run this for real, you'll use
src/data_acquisition.py with actual FTSE data instead.
"""

import numpy as np
import pandas as pd


def make_synthetic_universe(n_tickers=25, n_days=1500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    frames = []
    for i in range(n_tickers):
        ticker = f"SYN{i:02d}.L"
        # Each stock has a latent "quality" factor that mildly predicts drift,
        # simulating a faint, learnable alpha signal buried in noise.
        quality = rng.normal(0, 1)
        drift = 0.0002 + 0.0001 * quality
        vol = rng.uniform(0.012, 0.025)

        rets = rng.normal(drift, vol, n_days)
        price = 100 * np.exp(np.cumsum(rets))

        high = price * (1 + rng.uniform(0, 0.01, n_days))
        low = price * (1 - rng.uniform(0, 0.01, n_days))
        open_ = price * (1 + rng.normal(0, 0.003, n_days))
        volume = rng.integers(1_000_000, 8_000_000, n_days).astype(float)

        df = pd.DataFrame({
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": volume,
            "ticker": ticker,
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = make_synthetic_universe()
    out_path = "/home/claude/uk-alpha-ml/data/raw/all_prices.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Synthetic dataset written: {df.shape[0]} rows, {df['ticker'].nunique()} tickers -> {out_path}")
