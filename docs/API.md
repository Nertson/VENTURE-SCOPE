# VENTURE-SCOPE API Documentation

**Version**: 2.0 (Temporal Validation)  
**Last Updated**: January 2025  
**Author**: Arthur Pillet

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Ingestion](#2-data-ingestion)
3. [Feature Engineering](#3-feature-engineering)
4. [Machine Learning](#4-machine-learning)
   - [Baseline Model](#41-baseline-model)
   - [Temporal Model](#42-temporal-model-prod)
   - [Model Comparison](#43-model-comparison)
5. [Prediction & Inference](#5-prediction--inference)

---

## 1. Introduction

The **VENTURE-SCOPE** API provides a modular Python interface for analyzing startup success. It is structured into three main components:
- **Ingest**: Loading and enriching raw Crunchbase data.
- **Features**: Calculating VC-specific KPIs (Burn Multiple, Rule of 40, etc.) and investment scores.
- **ML**: Training models, evaluating performance (temporal vs baseline), and running predictions.

---

## 2. Data Ingestion

Module: `src.venture_scope.ingest.loaders_enriched`

Handles loading raw CSVs (companies, rounds, investments) and merging them into a single enriched dataset.

### `load_enriched_startups`

```python
from venture_scope.ingest.loaders_enriched import load_enriched_startups

df = load_enriched_startups(
    data_dir: str | Path,
    filter_funded: bool = True,
    min_funding: float = 0,
    verbose: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `data_dir`: Directory containing `objects.csv`, `funding_rounds.csv`, `investments.csv`.
- `filter_funded`: If `True`, removes companies with $0 or unknown funding.
- `min_funding`: Minimum USD funding threshold to keep.
- `verbose`: Print progress logs.

**Returns:**
- A `pd.DataFrame` containing enriched startup data (stage, funding, investor counts).

---

## 3. Feature Engineering

Modules: `src.venture_scope.features.kpi`, `src.venture_scope.features.scoring`

### 3.1 KPI Calculation

Calculates financial metrics mostly estimated from funding history and benchmarks.

#### `calculate_all_kpis`

```python
from venture_scope.features.kpi import calculate_all_kpis

df_kpis = calculate_all_kpis(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame
```

**Adds columns:**
- `estimated_revenue`: Annual revenue estimated from funding stage benchmarks.
- `capital_efficiency`: Revenue / Total Funding.
- `monthly_burn`: Estimated monthly burn rate.
- `runway_months`: Cash / Monthly Burn.
- `burn_multiple`: Annual Burn / Annual Revenue.
- `traction_index`: Composite growth score (0-100).
- `rule_of_40`: Growth rate + Profit margin estimator.

### 3.2 Investment Scoring

Composite score (0-100) aggregating multiple KPIs.

#### `calculate_investment_score`

```python
from venture_scope.features.scoring import calculate_investment_score

df_scored = calculate_investment_score(
    df: pd.DataFrame, 
    weights: dict = None, 
    verbose: bool = True
) -> pd.DataFrame
```

**Default Weights:**
- Rule of 40: 25%
- Traction Index: 25%
- Capital Efficiency: 20%
- Burn Multiple: 15%
- Runway: 15%

#### `rank_startups`

```python
from venture_scope.features.scoring import rank_startups

df_ranked = rank_startups(df: pd.DataFrame) -> pd.DataFrame
```
Sorts the DataFrame by `investment_score`.

#### `score_breakdown`

```python
from venture_scope.features.scoring import score_breakdown

score_breakdown(df: pd.DataFrame, company_name: str) -> None
```
Prints a detailed console report explaining how the score was derived for a specific company.

---

## 4. Machine Learning

Module: `src.venture_scope.ml`

### 4.1 Baseline Model

File: `src/venture_scope/ml/model.py`

Standard ML pipeline using random split (subject to look-ahead bias). Useful for benchmarking.

#### `run_ml_pipeline`

```python
from venture_scope.ml.model import run_ml_pipeline

results = run_ml_pipeline(
    input_file: str = "data/processed/startups_scored.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict
```

**Returns dict with:**
- `'model'`: Trained `RandomForestClassifier`.
- `'metrics'`: Accuracy, Precision, Recall, F1, ROC-AUC.
- `'feature_importance'`: Top features.

### 4.2 Temporal Model (PROD)

File: `src/venture_scope/ml/model_temporal.py`

Rigorous training pipeline that uses strict time-based splits (Train: 2000-2010, Val: 2011, Test: 2012-2013) to eliminate look-ahead bias.

#### `TemporalModelTrainer`

```python
from venture_scope.ml.model_temporal import TemporalModelTrainer

trainer = TemporalModelTrainer(data_dir='data/processed')
trainer.load_temporal_splits()
trainer.prepare_all_splits()
trainer.train_model()
trainer.evaluate_all()
trainer.save_model()
```

**Key Methods:**
- `load_temporal_splits()`: Loads pre-split temporal CSVs.
- `train_model()`: Trains Random Forest on the 2000-2010 window.
- `evaluate_all()`: Computes metrics across all splits and checks for overfitting.

### 4.3 Model Comparison

File: `src/venture_scope/ml/model_comparison.py`

Formal comparison of multiple algorithms (RF, Gradient Boosting, SVM, Logistic Regression).

#### `compare_models`

```python
from venture_scope.ml.model_comparison import compare_models

df_results = compare_models(X_train, X_test, y_train, y_test)
```
Returns a DataFrame comparing Accuracy, Precision, Recall, and ROC-AUC across models.

---

## 5. Prediction & Inference

File: `src/venture_scope/ml/predict.py`

Interactive or programmatic interface for predicting success of new startups.

#### `predict_startup`

```python
from venture_scope.ml.predict import predict_startup

result = predict_startup(
    funding_amount=10_000_000,
    stage='Series A',
    sector='saas',
    country='USA',
    investors_count=5,
    founded_year=2020,
    cutoff_date=None,  # Optional: for historical simulation
    use_temporal=True  # Use the unbiased temporal model
)
```

**Returns dict:**
- `success_prob`: Probability of IPO/Acquisition.
- `kpis`: Calculated KPIs.
- `interpretation`: Text description of strengths/concerns.
- `shift_warning`: Warnings if input looks like 2025 data (vs 2013 training context).

#### `check_distribution_shift`

```python
from venture_scope.ml.predict import check_distribution_shift

warnings = check_distribution_shift(funding_amount=..., stage=...)
```
checks for inflation/market shifts (e.g., a $5M Series A was huge in 2013, average in 2025).
