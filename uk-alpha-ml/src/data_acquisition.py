"""
data_acquisition.py
--------------------
Downloads daily OHLCV data for a universe of UK (FTSE 100) stocks using yfinance,
and a benchmark index for comparison.

IMPORTANT: This script needs internet access to Yahoo Finance. Run it on your
own machine (not inside a locked-down sandbox) with:

    python src/data_acquisition.py

Output: one Parquet file per ticker in data/raw/, plus a combined long-format
Parquet file data/raw/all_prices.parquet.
"""

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# A representative slice of liquid FTSE 100 constituents (Yahoo Finance tickers
# use the .L suffix for the London Stock Exchange). Feel free to expand this to
# the full FTSE 100/250 list later -- start with a manageable universe.
FTSE_UNIVERSE = [
    "AZN.L",   # AstraZeneca
    "HSBA.L",  # HSBC
    "SHEL.L",  # Shell
    "ULVR.L",  # Unilever
    "BP.L",    # BP
    "GSK.L",   # GSK
    "DGE.L",   # Diageo
    "RIO.L",   # Rio Tinto
    "BATS.L",  # British American Tobacco
    "REL.L",   # RELX
    "LSEG.L",  # London Stock Exchange Group
    "NG.L",    # National Grid
    "GLEN.L",  # Glencore
    "AAL.L",   # Anglo American
    "BARC.L",  # Barclays
    "LLOY.L",  # Lloyds Banking Group
    "VOD.L",   # Vodafone
    "PRU.L",   # Prudential
    "CPG.L",   # Compass Group
    "TSCO.L",  # Tesco
    "IMB.L",   # Imperial Brands
    "NWG.L",   # NatWest Group
    "STAN.L",  # Standard Chartered
    "AV.L",    # Aviva
    "SSE.L",   # SSE
    "RKT.L",   # Reckitt Benckiser
    "BT-A.L",  # BT Group
    "SGE.L",   # Sage Group
    "SMIN.L",  # Smiths Group
    "WTB.L",   # Whitbread
]

BENCHMARK = "^FTSE"  # FTSE 100 index

START_DATE = "2014-01-01"
END_DATE = None  # None = up to today


def download_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Download OHLCV history for a single ticker, return tidy DataFrame or None on failure."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            log.warning(f"No data returned for {ticker}, skipping.")
            return None
        df = df.reset_index()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        df["ticker"] = ticker
        return df
    except Exception as e:
        log.warning(f"Failed to download {ticker}: {e}")
        return None


def main():
    all_frames = []

    log.info(f"Downloading benchmark: {BENCHMARK}")
    bench = download_ticker(BENCHMARK, START_DATE, END_DATE)
    if bench is not None:
        bench.to_parquet(RAW_DIR / "benchmark_ftse100.parquet", index=False)
        log.info(f"Saved benchmark: {len(bench)} rows")

    for ticker in FTSE_UNIVERSE:
        log.info(f"Downloading {ticker} ...")
        df = download_ticker(ticker, START_DATE, END_DATE)
        if df is not None:
            out_path = RAW_DIR / f"{ticker.replace('.', '_')}.parquet"
            df.to_parquet(out_path, index=False)
            all_frames.append(df)
            log.info(f"  -> {len(df)} rows saved to {out_path.name}")
        time.sleep(0.5)  # be polite to the API

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(RAW_DIR / "all_prices.parquet", index=False)
        log.info(f"Combined dataset: {combined.shape[0]} rows, "
                 f"{combined['ticker'].nunique()} tickers -> all_prices.parquet")
    else:
        log.error("No data downloaded successfully. Check your internet connection.")


if __name__ == "__main__":
    main()
