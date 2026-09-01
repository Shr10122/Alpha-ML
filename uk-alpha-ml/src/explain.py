"""
explain.py
----------
SHAP feature importance analysis -- sanity-checks that the model isn't
relying on something spurious (e.g. a feature that's a proxy for calendar
time, or noise that happened to correlate in-sample).

Usage:
    python -m src.explain
"""

from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.models import prepare_binary_dataset, train_final_model, get_feature_columns

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feature_cols = get_feature_columns(feat)
    binary_df = prepare_binary_dataset(feat)

    model = train_final_model(binary_df, "lightgbm", feature_cols)

    sample = binary_df.dropna(subset=feature_cols).sample(
        n=min(2000, len(binary_df)), random_state=42
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[feature_cols])

    plt.figure()
    shap.summary_plot(shap_values, sample[feature_cols], show=False, plot_type="bar")
    plt.tight_layout()
    out_path = RESULTS_DIR / "shap_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved SHAP feature importance plot to {out_path}")


if __name__ == "__main__":
    main()
