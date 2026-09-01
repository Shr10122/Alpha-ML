"""
models.py
---------
Trains and evaluates classification models that predict whether a stock will
be a cross-sectional top or bottom performer over the next `horizon` days.

We frame this as BINARY classification: top-quantile (label 1) vs
bottom-quantile (label 0), dropping the noisy middle quantiles. This is a
standard quant ML simplification -- separating clear winners from clear
losers is more learnable than predicting the exact return.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from .cv import PurgedEmbargoedTimeSeriesSplit


FEATURE_PREFIX = "feat_"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def prepare_binary_dataset(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Keeps only the top quantile (label=1) and bottom quantile (label=0) rows,
    based on the pre-computed 'target_class' column (0..n_quantiles-1).
    """
    df = df.dropna(subset=["target_class"]).copy()
    top_q = n_quantiles - 1
    bottom_q = 0
    df = df[df["target_class"].isin([top_q, bottom_q])].copy()
    df["label"] = (df["target_class"] == top_q).astype(int)
    return df


MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=0.5),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=50, random_state=42, n_jobs=-1
    ),
    "lightgbm": lambda: lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    ),
}


@dataclass
class FoldResult:
    model_name: str
    fold: int
    auc: float
    n_train: int
    n_test: int


def walk_forward_evaluate(
    df: pd.DataFrame,
    model_names: list[str],
    n_splits: int = 5,
    horizon: int = 5,
    embargo: int = 5,
) -> pd.DataFrame:
    """
    Runs purged/embargoed walk-forward CV for each model, returns a tidy
    dataframe of per-fold AUC scores.
    """
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)

    splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=n_splits, horizon=horizon, embargo=embargo)
    results = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(df["Date"])):
        X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, "label"]
        X_test, y_test = df.loc[test_idx, feature_cols], df.loc[test_idx, "label"]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        for name in model_names:
            model = MODEL_REGISTRY[name]()
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
            results.append(FoldResult(name, fold_i, auc, len(train_idx), len(test_idx)))

    return pd.DataFrame([r.__dict__ for r in results])


def train_final_model(df: pd.DataFrame, model_name: str, feature_cols: list[str]):
    """Train on ALL available (non-null) data -- used for generating live/backtest signals."""
    clean = df.dropna(subset=feature_cols + ["label"])
    model = MODEL_REGISTRY[model_name]()
    model.fit(clean[feature_cols], clean["label"])
    return model


def generate_full_predictions(df: pd.DataFrame, model, feature_cols: list[str]) -> pd.Series:
    """
    Generates predicted probability-of-being-top-performer for EVERY row with
    complete features (not just the binary-filtered subset) -- this is what
    we feed into the backtest, since the strategy needs a score for every
    stock on every day, not just historical extreme movers.
    """
    mask = df[feature_cols].notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    preds.loc[mask] = model.predict_proba(df.loc[mask, feature_cols])[:, 1]
    return preds
