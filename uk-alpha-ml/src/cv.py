"""
cv.py
-----
Purged, embargoed time-series cross-validation.

Why not sklearn's TimeSeriesSplit alone? Because our target is a FORWARD
return over `horizon` days. That means the label for row t is only fully
known at t + horizon. If a test fold starts right after a train fold ends,
some training labels will have been computed using price data that overlaps
with the test period -> leakage. We fix this with:

  1. PURGING: remove training samples whose label window overlaps the test window.
  2. EMBARGO: additionally remove a small buffer of training samples right
     after the test window, since the model's own recently-seen test-adjacent
     info could otherwise leak backwards via feature autocorrelation.

Reference concept: Marcos Lopez de Prado, "Advances in Financial Machine
Learning", Chapter 7 (Cross-Validation in Finance).
"""

import numpy as np
import pandas as pd


class PurgedEmbargoedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo: int = 5):
        """
        n_splits: number of walk-forward folds.
        horizon: forward-return horizon used to build the target (in trading days).
        embargo: extra trading days to drop after each test fold.
        """
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, dates: pd.Series):
        """
        dates: a Series (aligned to the dataframe's row order) of trading dates
               (can have duplicates, since multiple tickers share a date).
        Yields (train_idx, test_idx) numpy index arrays into the original dataframe.
        """
        unique_dates = np.sort(dates.unique())
        n = len(unique_dates)
        fold_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start_i = fold_bounds[i]
            test_end_i = fold_bounds[i + 1]
            if test_start_i == test_end_i:
                continue

            test_dates = unique_dates[test_start_i:test_end_i]
            test_start_date, test_end_date = test_dates[0], test_dates[-1]

            test_mask = dates.isin(test_dates)

            # Purge: drop training rows whose label window [date, date+horizon]
            # overlaps the test window.
            purge_lower = test_start_date - np.timedelta64(self.horizon, "D")
            purge_upper = test_end_date

            # Embargo: also drop a buffer of rows right after the test window.
            embargo_upper = test_end_date + np.timedelta64(self.embargo, "D")

            train_mask = ~test_mask & ~(
                (dates >= purge_lower) & (dates <= embargo_upper)
            )

            train_idx = np.where(train_mask.values)[0]
            test_idx = np.where(test_mask.values)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx
