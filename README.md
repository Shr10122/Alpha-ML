# Machine Learning for Alpha Generation & Backtesting (UK Equities)

A cross-sectional long-short equity strategy on FTSE 100 stocks: ML models
rank stocks by predicted relative forward return, the top decile is bought
and the bottom decile shorted, and the resulting strategy is backtested with
explicit controls against the biases that make most amateur quant backtests
worthless (look-ahead bias, leakage across CV folds, and overlapping-window
double counting).

## Why this project

Built as a portfolio project while starting an MSc in Data Science, to
demonstrate applied ML skills in a quant/finance context: feature
engineering on time series, leakage-safe cross-validation, and the
methodological rigor that separates a real research process from a
p-hacked backtest.

## Methodology summary

1. **Data**: daily OHLCV for ~30 liquid FTSE 100 constituents + FTSE 100
   index benchmark, via `yfinance` (2014-present).
2. **Features**: momentum (5/10/20/60-day returns, MA ratios, RSI, MACD),
   volatility (rolling std, ATR proxy), mean-reversion (price z-score),
   volume (volume z-score, OBV change), and cross-sectional percentile
   ranks. All features use only information available up to time *t*.
3. **Target**: forward return over a 5-trading-day horizon, converted to a
   cross-sectional quintile label (binary top-vs-bottom classification).
4. **Cross-validation**: custom **purged, embargoed walk-forward CV**
   (concept from López de Prado, *Advances in Financial Machine Learning*)
   to prevent the forward-looking target window from leaking across
   train/test splits.
5. **Models compared**: Logistic Regression, Random Forest, LightGBM —
   selected by mean out-of-sample AUC across folds.
6. **Backtest**: a from-scratch vectorized long-short backtester (no
   external backtesting library — every calculation is auditable),
   rebalanced on non-overlapping dates, with a simple proportional
   transaction cost model. Reports Sharpe, Sortino, max drawdown, CAGR,
   hit rate.
7. **Explainability**: SHAP feature importance to sanity-check the model
   isn't relying on something spurious.

## A methodological trap I hit (and fixed) while building this

The 5-day forward return target is recomputed on *every single day*, so
naively rebalancing daily makes consecutive "trading days" overlap by 4 of
their 5 underlying return days. Compounding those overlapping returns daily
produced an initial backtest showing **~970% total return** — which is the
exact "too good to be true" signature of a broken backtest, not a working
strategy. The fix: only rebalance on **non-overlapping** dates, spaced by
the forward horizon. This is documented in `src/run_pipeline.py` and is
worth understanding deeply — it's one of the most common ways beginner
quant projects fool their authors.

## Project structure

```
uk-alpha-ml/
├── data/
│   ├── raw/              # raw OHLCV parquet files (gitignore these — regenerate locally)
│   └── processed/        # feature-engineered dataset
├── src/
│   ├── data_acquisition.py   # downloads FTSE 100 OHLCV via yfinance
│   ├── features.py           # feature engineering + target construction
│   ├── cv.py                 # purged/embargoed walk-forward CV
│   ├── models.py             # model training + walk-forward evaluation
│   ├── backtest.py           # custom vectorized long-short backtester
│   ├── explain.py            # SHAP feature importance
│   ├── report.py             # equity curve / drawdown plots
│   └── run_pipeline.py       # end-to-end orchestration
├── notebooks/
│   └── alpha_research.ipynb  # exploratory walkthrough with inline plots
├── tests/
│   └── make_synthetic_data.py  # synthetic data generator for pipeline smoke-testing
├── results/               # generated metrics, plots, CV results
├── app.py                 # interactive Streamlit dashboard
├── Dockerfile              # container image for deployment
├── .dockerignore
└── requirements.txt
```

## Visual diagnostics (`python -m src.report`)

Generates 6 plots into `results/`:

1. Equity curve + drawdown
2. Rolling Sharpe ratio (spot regime shifts in the signal over time)
3. Long leg vs short leg contribution (which side is actually driving P&L)
4. Monthly returns calendar heatmap
5. Feature correlation matrix (checks for redundant/collinear features)
6. Walk-forward CV AUC by fold and model

## Interactive dashboard (`streamlit run app.py`)

A Plotly-based Streamlit dashboard covering the same diagnostics
interactively, plus a feature explorer to inspect any engineered feature
over time for any ticker. This is also the app that gets containerized and
deployed to AWS below.

```bash
streamlit run app.py
```

## Deploying to AWS (EC2 + Docker)

1. **Build and test the Docker image locally first**:
   ```bash
   docker build -t uk-alpha-ml .
   docker run -p 8501:8501 uk-alpha-ml
   # visit http://localhost:8501
   ```
2. **Launch an EC2 instance** (Ubuntu 22.04, t2.micro/t3.micro is enough for
   a demo), open inbound port 8501 (or 80, if you set up a reverse proxy) in
   its security group.
3. **Install Docker on the instance**, copy the project over (`scp` or
   `git clone` if you push this to GitHub), then `docker build` and
   `docker run` exactly as you did locally.
4. Visit `http://<your-ec2-public-ip>:8501`.

See the full step-by-step walkthrough Claude can provide interactively, or
the "Possible extensions" section below for the next step up (ECS/App
Runner, HTTPS via a reverse proxy + domain).

## How to run (with real data)

```bash
pip install -r requirements.txt

# Step 1: download real FTSE 100 data (needs internet access)
python src/data_acquisition.py

# Step 2: run the full pipeline (features -> CV -> model -> backtest)
python -m src.run_pipeline

# Step 3 (optional): generate plots and SHAP explainability
python -m src.report
python -m src.explain
```

Or open `notebooks/alpha_research.ipynb` for an interactive walkthrough.

## Smoke-testing without internet access

`tests/make_synthetic_data.py` generates a synthetic multi-stock dataset
with a deliberately faint, learnable signal, so you can validate the whole
pipeline runs correctly before spending time on real data:

```bash
python tests/make_synthetic_data.py
python -m src.run_pipeline
```

## Honest limitations

- **Survivorship bias**: only current FTSE 100 constituents are used;
  companies that were delisted or went bankrupt during the sample period
  are excluded, which likely flatters results slightly.
- **Transaction costs** are a simplified flat-bps-per-turnover estimate,
  not a real market-impact model.
- **Execution assumption**: trades are assumed to fill at the same closing
  price used to compute returns, ignoring slippage.
- **Small AUCs are expected and correct**: walk-forward AUCs around
  0.51-0.53 are typical for daily equity return prediction. A much higher
  AUC on this kind of data would indicate leakage, not skill.
- This project is for research/educational purposes only and is not
  investment advice.

## Possible extensions

- Expand universe to full FTSE 100/250.
- Add fundamental factors (P/E, dividend yield) via a fundamentals data
  source.
- Walk-forward retraining (currently trains once on all data for the final
  model) to test regime robustness explicitly.
- Position sizing via volatility targeting instead of equal-weight deciles.
