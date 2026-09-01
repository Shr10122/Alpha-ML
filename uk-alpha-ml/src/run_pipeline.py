"""
run_pipeline.py
----------------
End-to-end orchestration: load raw prices -> build features & targets ->
walk-forward CV comparison of models -> train final model -> generate
predictions -> backtest -> print/save performance report.

Usage:
    python -m src.run_pipeline
"""

import json
from pathlib import Path

import pandas as pd

from src.features import (
    build_features_all,
    add_cross_sectional_rank,
    make_classification_target,
)
from src.models import (
    prepare_binary_dataset,
    walk_forward_evaluate,
    train_final_model,
    generate_full_predictions,
    get_feature_columns,
)
from src.backtest import build_portfolio_weights, run_backtest, performance_metrics

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "all_prices.parquet"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORWARD_HORIZON = 5   # trading days
N_QUANTILES = 5
N_CV_SPLITS = 5
EMBARGO_DAYS = 5
MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]
PERIODS_PER_YEAR = 252 / FORWARD_HORIZON  # number of non-overlapping rebalances/year


def main():
    print("=" * 70)
    print("STEP 1: Loading raw price data")
    print("=" * 70)
    raw = pd.read_parquet(RAW_PATH)
    raw["Date"] = pd.to_datetime(raw["Date"])
    print(f"Loaded {raw.shape[0]} rows, {raw['ticker'].nunique()} tickers, "
          f"{raw['Date'].min().date()} -> {raw['Date'].max().date()}")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering + target construction")
    print("=" * 70)
    feat = build_features_all(raw, forward_horizon=FORWARD_HORIZON)
    feat = add_cross_sectional_rank(feat, "feat_return_20d", "feat_return_20d_rank")
    feat = add_cross_sectional_rank(feat, "feat_vol_20d", "feat_vol_20d_rank")
    feat = make_classification_target(feat, n_quantiles=N_QUANTILES)
    feat.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    feature_cols = get_feature_columns(feat)
    print(f"Built {len(feature_cols)} features: {feature_cols}")
    print(f"Rows with usable target: {feat['target_class'].notna().sum()} / {len(feat)}")

    print("\n" + "=" * 70)
    print("STEP 3: Purged/embargoed walk-forward CV -- comparing models")
    print("=" * 70)
    binary_df = prepare_binary_dataset(feat, n_quantiles=N_QUANTILES)
    cv_results = walk_forward_evaluate(
        binary_df, MODEL_NAMES, n_splits=N_CV_SPLITS,
        horizon=FORWARD_HORIZON, embargo=EMBARGO_DAYS,
    )
    cv_results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    print(cv_results.groupby("model_name")["auc"].agg(["mean", "std", "count"]))

    best_model_name = cv_results.groupby("model_name")["auc"].mean().idxmax()
    print(f"\nBest model by mean walk-forward AUC: {best_model_name}")

    print("\n" + "=" * 70)
    print("STEP 4: Train final model on full binary dataset")
    print("=" * 70)
    final_model = train_final_model(binary_df, best_model_name, feature_cols)

    print("\n" + "=" * 70)
    print("STEP 5: Generate scores for ALL stocks/dates (for backtest)")
    print("=" * 70)
    feat["pred_score"] = generate_full_predictions(feat, final_model, feature_cols)
    print(f"Generated scores for {feat['pred_score'].notna().sum()} rows")

    print("\n" + "=" * 70)
    print("STEP 6: Build portfolio weights + run backtest")
    print("=" * 70)
    scored = feat.dropna(subset=["pred_score", "target_fwd_return"])

    # IMPORTANT: target_fwd_return is a FORWARD_HORIZON-day return, recomputed
    # on every single trading day. If we rebalanced on every day, consecutive
    # rebalance dates would overlap by (FORWARD_HORIZON - 1) days of the same
    # underlying price moves, and naively compounding those daily "returns"
    # massively double-counts the same market moves -- this is what produced
    # an unrealistic ~970% backtest return on the first run below. The fix:
    # only rebalance on NON-OVERLAPPING dates, spaced FORWARD_HORIZON days apart.
    rebalance_dates = sorted(scored["Date"].unique())[::FORWARD_HORIZON]
    scored = scored[scored["Date"].isin(rebalance_dates)]
    print(f"Using {len(rebalance_dates)} non-overlapping rebalance dates "
          f"(every {FORWARD_HORIZON} trading days) to avoid double-counting overlapping return windows")

    weighted = build_portfolio_weights(scored, score_col="pred_score", top_frac=0.15, bottom_frac=0.15)
    bt_result = run_backtest(weighted, forward_return_col="target_fwd_return", holding_period=FORWARD_HORIZON)
    bt_result.to_csv(RESULTS_DIR / "backtest_returns.csv", index=False)

    metrics = performance_metrics(bt_result, periods_per_year=PERIODS_PER_YEAR)
    print("\nBacktest performance (long-short, 15%/15%, net of estimated costs):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(RESULTS_DIR / "performance_summary.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, float) else v) for k, v in metrics.items()}, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
