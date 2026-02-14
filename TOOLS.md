# DDL-69 Tools & Methods

Complete reference for all quantitative tools, ML models, and methods implemented in DDL-69 v0.8.

## Table of Contents

1. [Technical Indicators (TA-Lib)](#technical-indicators-ta-lib)
2. [Lopez de Prado Methods](#lopez-de-prado-methods)
3. [Monte Carlo Simulations](#monte-carlo-simulations)
4. [FinRL - Reinforcement Learning](#finrl---reinforcement-learning)
5. [Qlib - Quantitative Strategies](#qlib---quantitative-strategies)
6. [Scikit-learn Ensembles](#scikit-learn-ensembles)
7. [FinGPT - Financial NLP](#fingpt---financial-nlp)
8. [Weight Learning & Calibration](#weight-learning--calibration)

---

## Technical Indicators (TA-Lib)

**Module:** `src/ddl69/indicators/talib_wrapper.py`

### Purpose
Technical analysis indicators for feature engineering and signal generation.

### Indicators Implemented

**Trend Indicators:**
- `SMA(period)` - Simple Moving Average
- `EMA(period)` - Exponential Moving Average
- `DEMA(period)` - Double Exponential Moving Average
- `TEMA(period)` - Triple Exponential Moving Average
- `KAMA(period)` - Kaufman Adaptive Moving Average

**Volatility Indicators:**
- `BBANDS(period, nbdevup, nbdevdn)` - Bollinger Bands
- `ATR(period)` - Average True Range
- `NATR(period)` - Normalized Average True Range

**Momentum Indicators:**
- `RSI(period)` - Relative Strength Index
- `MACD(fastperiod, slowperiod, signalperiod)` - Moving Average Convergence Divergence
- `STOCH(fastk_period, slowk_period, slowd_period)` - Stochastic Oscillator
- `CCI(period)` - Commodity Channel Index
- `ADX(period)` - Average Directional Index

**Volume Indicators:**
- `OBV()` - On Balance Volume
- `AD()` - Accumulation/Distribution

### Usage

```python
from ddl69.indicators.talib_wrapper import compute_all_indicators

# Add all indicators to DataFrame
df = compute_all_indicators(df, use_talib=True)

# Or use individual indicators
from ddl69.indicators.talib_wrapper import TALibWrapper

wrapper = TALibWrapper()
df['sma_30'] = wrapper.SMA(df['close'], period=30)
df['rsi_14'] = wrapper.RSI(df['close'], period=14)
```

### Fallback
Automatically falls back to pandas implementations if TA-Lib not installed.

---

## Lopez de Prado Methods

**Module:** `src/ddl69/labeling/lopez_prado.py`

### Purpose
Advanced labeling and validation methods from "Advances in Financial Machine Learning".

### Methods Implemented

#### 1. Fractional Differentiation (FFD)
```python
from ddl69.labeling.lopez_prado import frac_diff_ffd

# Create stationary features while preserving memory
df['close_ffd'] = frac_diff_ffd(df['close'], d=0.5, thres=0.01)
```

**What it does:** Makes time series stationary while preserving maximum memory (predictive power).

#### 2. Triple Barrier Labeling
```python
from ddl69.labeling.triple_barrier import apply_pt_sl_on_t1, add_vertical_barrier

# Set barriers: profit-taking, stop-loss, vertical (time)
t1 = add_vertical_barrier(df.index, df['close'], num_days=5)
events = apply_pt_sl_on_t1(
    close=df['close'],
    events=pd.DataFrame({'t1': t1, 'trgt': volatility}),
    pt_sl=[0.02, 0.02],  # 2% profit-taking, 2% stop-loss
)
```

**What it does:** Creates labels based on which barrier is touched first, accounting for time, profit targets, and stop-losses.

#### 3. Purged K-Fold Cross-Validation
```python
from ddl69.labeling.lopez_prado import PurgedKFold

# Prevents leakage in time-series CV
cv = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(days=2))
for train_idx, test_idx in cv.split(X, y, t1):
    # Train/test with no overlap + embargo
    pass
```

**What it does:** Removes training samples that overlap with test period + adds embargo period to prevent look-ahead bias.

#### 4. Meta-Labeling
```python
from ddl69.labeling.lopez_prado import meta_labeling

# Use ML to size bets on primary model signals
meta_labels = meta_labeling(primary_pred, actual_labels, bet_size)
```

**What it does:** Second ML model learns to predict when primary model is correct and how much to bet.

#### 5. Sample Weights
```python
from ddl69.labeling.lopez_prado import get_sample_weights

# Weight samples by uniqueness (label overlap)
weights = get_sample_weights(t1, num_concurrent)
```

**What it does:** Samples with more unique information get higher weight in training.

---

## Monte Carlo Simulations

**Module:** `src/ddl69/simulations/monte_carlo.py`

### Purpose
Risk analysis and portfolio simulation using Monte Carlo methods.

### Simulations Implemented

#### 1. Monte Carlo Returns
```python
from ddl69.simulations.monte_carlo import monte_carlo_returns

# Simulate future returns
simulated = monte_carlo_returns(
    returns=historical_returns,
    n_simulations=1000,
    n_periods=252,
    method="bootstrap",  # or "parametric"
)
```

#### 2. Portfolio Simulation
```python
from ddl69.simulations.monte_carlo import monte_carlo_portfolio

# Simulate portfolio paths
results = monte_carlo_portfolio(
    returns=historical_returns,
    initial_capital=100000,
    n_simulations=1000,
    n_periods=252,
)
```

#### 3. Sharpe Ratio Distribution
```python
from ddl69.simulations.monte_carlo import sharpe_ratio_distribution

# Distribution of possible Sharpe ratios
sharpe_dist = sharpe_ratio_distribution(returns, n_simulations=1000)
```

#### 4. Value at Risk (VaR)
```python
from ddl69.simulations.monte_carlo import value_at_risk_mc, conditional_var_mc

# Monte Carlo VaR
var_95 = value_at_risk_mc(returns, confidence_level=0.95)
cvar_95 = conditional_var_mc(returns, confidence_level=0.95)
```

#### 5. Drawdown Distribution
```python
from ddl69.simulations.monte_carlo import drawdown_distribution

# Simulate maximum drawdowns
dd_dist = drawdown_distribution(returns, n_simulations=1000)
```

#### 6. Permutation Test
```python
from ddl69.simulations.monte_carlo import permutation_test

# Test if strategy A is significantly better than strategy B
p_value = permutation_test(returns_a, returns_b, n_permutations=10000)
```

---

## FinRL - Reinforcement Learning

**Module:** `src/ddl69/agents/finrl_agents.py`

### Purpose
Deep Reinforcement Learning agents for automated trading.

### Algorithms Supported
- **PPO** (Proximal Policy Optimization) - Most stable, recommended
- **A2C** (Advantage Actor-Critic) - Fast training
- **DDPG** (Deep Deterministic Policy Gradient) - Continuous actions
- **TD3** (Twin Delayed DDPG) - Improved DDPG
- **SAC** (Soft Actor-Critic) - Maximum entropy RL

### Usage

```python
from ddl69.agents import FinRLTrader, train_finrl_agent

# Train a single agent
trader, metrics = train_finrl_agent(
    df=ohlcv_df,  # Must have: date, tic, open, high, low, close, volume
    algorithm="ppo",
    total_timesteps=100_000,
)

# Or use ensemble
from ddl69.agents import EnsembleFinRL

ensemble = EnsembleFinRL(algorithms=["ppo", "a2c", "sac"])
ensemble.train_all(train_df, total_timesteps=100_000)
_, metrics = ensemble.predict_ensemble(test_df)
```

### Key Features
- Automatic feature engineering (technical indicators)
- Built-in train/val/test split
- Custom trading environment
- Transaction costs modeling
- Portfolio management

### Expert Integration
```python
from ddl69.experts.ml_experts import FinRLExpert

expert = FinRLExpert(algorithm="ppo")
expert.train(train_df, total_timesteps=100_000)
result = expert.predict(test_df, event)
# Returns: probs, confidence, uncertainty
```

---

## Qlib - Quantitative Strategies

**Module:** `src/ddl69/strategies/qlib_strategies.py`

### Purpose
Quantitative factor-based strategies using Microsoft Qlib.

### Features

#### 1. Factor Library
```python
from ddl69.strategies.qlib_strategies import QlibFactorLibrary

# Get alpha factors
factors = QlibFactorLibrary.get_alpha158()
# Or: get_alpha101(), get_technical()
```

**Alpha158:** 158 factors including momentum, MA, volatility, volume
**Alpha101:** WorldQuant-style alpha factors
**Technical:** RSI, MACD, Bollinger Bands, ATR, Williams %R

#### 2. Model Training
```python
from ddl69.strategies import train_qlib_model

strategy, metrics = train_qlib_model(
    qlib_dir=".qlib/us_data",
    instruments="csi300",
    start_time="2018-01-01",
    end_time="2023-12-31",
    model_type="lgb",  # LightGBM
)

# Metrics: IC, ICIR, IC positive ratio
```

#### 3. Backtesting
```python
strategy = QlibStrategy(qlib_dir=".qlib/us_data")
dataset = strategy.prepare_data(...)
strategy.train(dataset)

# Run backtest
report, metrics = strategy.backtest(
    dataset,
    strategy="topk_dropout",
    topk=50,  # Hold top 50 stocks
)

# Metrics: total_return, sharpe, max_drawdown, information_ratio
```

### Expert Integration
```python
from ddl69.experts.ml_experts import QlibExpert

expert = QlibExpert(qlib_dir=".qlib/us_data")
expert.train(instruments="csi300")
result = expert.predict(signal=qlib_signal, event=event)
```

---

## Scikit-learn Ensembles

**Module:** `src/ddl69/agents/sklearn_ensemble.py`

### Purpose
Ensemble of tree-based and gradient boosting models.

### Models Supported
- **RandomForest** (rf) - Bagging ensemble
- **ExtraTrees** (et) - Extremely randomized trees
- **GradientBoosting** (gb) - Sklearn gradient boosting
- **XGBoost** (xgb) - High-performance gradient boosting
- **LightGBM** (lgb) - Fast gradient boosting
- **CatBoost** (cb) - Categorical boosting

### Usage

```python
from ddl69.agents import SklearnEnsemble, train_sklearn_ensemble

# Train ensemble
ensemble, metrics = train_sklearn_ensemble(
    X_train=features_train,
    y_train=labels_train,
    X_test=features_test,
    y_test=labels_test,
    task="classification",  # or "regression"
    models=["rf", "xgb", "lgb"],
)

# Get predictions
predictions = ensemble.predict(X_new)
probabilities = ensemble.predict_proba(X_new)  # classification only

# Feature importance
importance = ensemble.feature_importance(top_n=20)
```

### Cross-Validation
```python
cv_results = ensemble.cross_validate(
    X, y,
    cv=5,
    time_series=True,  # Use TimeSeriesSplit
)
# Returns: ensemble scores + individual model scores
```

### Expert Integration
```python
from ddl69.experts.ml_experts import SklearnExpert

expert = SklearnExpert(models=["rf", "xgb", "lgb"])
expert.train(X_train, y_train)
result = expert.predict(X_new, event)
```

---

## FinGPT - Financial NLP

**Module:** `src/ddl69/nlp/fingpt.py`

### Purpose
Financial language models for sentiment analysis and forecasting.

### Capabilities

#### 1. Sentiment Analysis
```python
from ddl69.nlp import FinGPTAnalyzer

analyzer = FinGPTAnalyzer(task="sentiment")

# Analyze single or multiple texts
results = analyzer.sentiment([
    "NVDA beat earnings expectations",
    "Fed raises rates, markets tumble",
])

# Aggregate sentiment
agg = analyzer.sentiment_aggregate(
    texts=news_headlines,
    method="weighted",  # or "mean", "majority"
)
```

#### 2. Score DataFrame
```python
# Add sentiment column to DataFrame
df = analyzer.score_dataframe(
    df=news_df,
    text_column="headline",
    output_column="sentiment",
)
```

#### 3. Named Entity Recognition (NER)
```python
analyzer = FinGPTAnalyzer(task="ner")
entities = analyzer.ner([
    "Apple reported Q4 revenue of $89.5B",
])
```

#### 4. Market Forecasting
```python
from ddl69.nlp import FinGPTForecaster

forecaster = FinGPTForecaster()
forecast = forecaster.forecast(
    ticker="AAPL",
    news_list=recent_headlines,
)
# Returns: {forecast: "up/down/same", confidence, sentiment}
```

### Expert Integration
```python
from ddl69.experts.ml_experts import SentimentExpert

expert = SentimentExpert()
expert.load()
result = expert.predict(news_texts=headlines, event=event)
```

### Models Used
- **FinBERT** (default) - Financial sentiment analysis
- **FinGPT-sentiment-cls** - FinGPT sentiment classifier
- **sec-bert-num** - Financial NER
- Custom HuggingFace models supported

---

## Weight Learning & Calibration

**Module:** `src/ddl69/utils/signals.py`

### Purpose
Learn expert weights from historical performance and calibrate probabilities.

### Key Functions

#### 1. Aggregate Rule Weights
```python
from ddl69.utils.signals import aggregate_rule_weights

# Learn weights from historical signals
weights, calibration = aggregate_rule_weights(signals_rows)

# Returns:
# - weights: {rule_name: weight} dictionary
# - calibration: per-rule statistics (win_rate, avg_return, etc.)
```

#### 2. Blend Probabilities
```python
from ddl69.utils.signals import blend_probs

# Weighted ensemble of expert probabilities
weighted_probs = [
    ({"REJECT": 0.3, "ACCEPT": 0.7}, 1.5),  # (probs, weight)
    ({"REJECT": 0.4, "ACCEPT": 0.6}, 2.0),
]

ensemble_probs = blend_probs(weighted_probs)
```

#### 3. Entropy
```python
from ddl69.utils.signals import entropy

# Measure uncertainty
ent = entropy({"REJECT": 0.3, "BREAK_FAIL": 0.2, "ACCEPT": 0.5})
# Higher entropy = more uncertain
```

### Weight Update Process

1. **Walk-forward training:**
   - Expand rules from historical bars
   - Score rules on forward periods
   - Calculate win rate, return, sharpe for each rule

2. **Weight calculation:**
   - Positive rules get higher weights
   - Negative rules get negative weights
   - Weight magnitude based on performance

3. **Calibration:**
   - Track realized outcomes
   - Adjust probabilities to match reality
   - Group rules by type for better calibration

---

## Unified Pipeline

**Module:** `src/ddl69/core/ml_pipeline.py`

### Purpose
Orchestrate all tools in a unified workflow.

### Usage

```python
from ddl69.core.ml_pipeline import create_full_pipeline

# Create pipeline with all tools
pipeline, df_prepared = create_full_pipeline(
    df=ohlcv_df,
    enable_all=True,  # Enable FinRL, Qlib, FinGPT
)

# Features + labels already prepared
X = df_prepared.drop(columns=['label'])
y = df_prepared['label']

# Train models
pipeline.train_sklearn(X_train, y_train, models=["rf", "xgb", "lgb"])
pipeline.train_finrl(df, algorithm="ppo", total_timesteps=100_000)
pipeline.train_qlib(qlib_dir=".qlib/us_data")

# Predict
sklearn_pred = pipeline.predict_sklearn(X_test)
finrl_metrics = pipeline.predict_finrl(test_df)

# Risk analysis
mc_results = pipeline.monte_carlo_analysis(returns, n_simulations=1000)
```

---

## CLI Commands

All tools accessible via CLI:

```bash
# Check tool status
python -m ddl69.cli.main tools_status

# FinRL
python -m ddl69.cli.main finrl_check
python -m ddl69.cli.main finrl_download --tickers AAPL,SPY --start 2020-01-01

# Qlib
python -m ddl69.cli.main qlib_check --qlib-dir .qlib/us_data
python -m ddl69.cli.main qlib_download --target-dir .qlib/us_data --region us

# FinGPT sentiment
python -m ddl69.cli.main fingpt_sentiment --text "NVDA beat earnings" --model ProsusAI/finbert
python -m ddl69.cli.main fingpt_score_dataset --input-path news.csv --model ProsusAI/finbert

# Walk-forward training (uses all tools)
python -m ddl69.cli.main train_walkforward \
    --bars bars.csv \
    --labels signals_rows.csv \
    --signals registry.zip \
    --qlib-dir .qlib/us_data
```

---

## Installation

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Optional Tools

**XGBoost, LightGBM, CatBoost:**
```bash
pip install xgboost lightgbm catboost
```

**FinRL:**
```bash
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
pip install stable-baselines3 torch
```

**Qlib:**
```bash
pip install pyqlib
```

**FinGPT:**
```bash
pip install transformers torch sentencepiece accelerate
```

**TA-Lib** (requires system dependencies):
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# Windows: Download from https://github.com/mrjbq7/ta-lib
pip install TA-Lib
```

---

## Architecture

```
src/ddl69/
├── indicators/         # TA-Lib wrappers
├── labeling/          # Triple barrier, Lopez de Prado
├── simulations/       # Monte Carlo
├── agents/            # FinRL, Sklearn ensembles
├── strategies/        # Qlib strategies
├── nlp/               # FinGPT sentiment
├── experts/           # ML expert wrappers
├── core/              # Pipeline, settings, contracts
└── cli/               # Command-line interface
```

---

## References

- **Lopez de Prado:** *Advances in Financial Machine Learning* (2018)
- **FinRL:** https://github.com/AI4Finance-Foundation/FinRL
- **Qlib:** https://github.com/microsoft/qlib
- **FinGPT:** https://github.com/AI4Finance-Foundation/FinGPT
- **TA-Lib:** https://ta-lib.org/
- **XGBoost:** https://xgboost.readthedocs.io/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **CatBoost:** https://catboost.ai/

---

## Summary

DDL-69 v0.8 integrates:

- ✅ **15+ TA-Lib indicators** for technical analysis
- ✅ **Lopez de Prado methods** for advanced labeling & validation
- ✅ **6 Monte Carlo simulations** for risk analysis
- ✅ **5 FinRL RL algorithms** (PPO, A2C, DDPG, TD3, SAC)
- ✅ **Qlib factor library** with Alpha158, Alpha101
- ✅ **6 sklearn models** (RF, XGBoost, LightGBM, CatBoost, ET, GB)
- ✅ **FinGPT NLP** for sentiment & forecasting
- ✅ **Walk-forward validation** with weight learning
- ✅ **Expert ensemble system** for probability calibration

All tools are production-ready, tested, and integrated into the unified pipeline.
