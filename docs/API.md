# VENTURE-SCOPE API Documentation

**Version**: 2.0 (Temporal Validation)  
**Last Updated**: January 2025  
**Author**: Arthur Pillet

---

## Table of Contents

1. Installation
2. Quick Start
3. Core Functions
4. Data Processing
5. Model Training & Prediction
6. Analysis Scripts
7. Testing
8. CLI Tools
9. File Structure

---

## 1. Installation

### Requirements

- Python 3.8.2+
- pip 20.0+
- Conda (recommended)

### Environment Setup
```bash
# Clone repository
git clone [repository-url]
cd VENTURE-SCOPE

# Create conda environment
conda env create -f environment.yml
conda activate venture-scope

# Install package in development mode
pip install -e .
```

### Verify Installation
```bash
# Run tests
pytest tests/  # Should pass 91/92 tests

# Check imports
python -c "from venture_scope.data import preprocessing; print('OK')"
```

---

## 2. Quick Start

### Temporal Split + Train + Evaluate
```bash
# 1. Create temporal splits
python scripts/temporal_split.py

# 2. Train temporal model (5 min)
python src/venture_scope/ml/model_temporal.py

# 3. Generate all analyses (10 min)
python scripts/compare_models.py
python scripts/error_analysis.py
python scripts/distribution_shift_analysis.py

# 4. Compare results with old models  with look ahead bias
python src/venture_scope/ml/model.py
python src/venture_scope/ml/model_comparison.py


# 5. Try the prediction tool 
python src/venture_scope/ml/predict.py


```

### Programmatic Usage
```python
from venture_scope.data.preprocessing import load_temporal_data
from venture_scope.ml.model_temporal import train_temporal_model
from venture_scope.ml.evaluation import evaluate_model

# Load temporal splits
train, val, test = load_temporal_data()

# Train model
model = train_temporal_model(train, val)

# Evaluate
metrics = evaluate_model(model, test)
print(f"Recall: {metrics['recall']:.3f}")  # 0.938
```

---

## 3. Core Functions

### 3.1 Data Loading

#### load_startups()
```python
from venture_scope.data.preprocessing import load_startups

df = load_startups(
    filter_funded=True,
    min_funding=0,
    include_international=True
)
```

**Parameters**:
- `filter_funded` (bool, default=True): Keep only companies with funding_total_usd > 0
- `min_funding` (float, default=0): Minimum funding threshold in USD
- `include_international` (bool, default=True): Include non-USA companies

**Returns**:
- `pandas.DataFrame` with columns:
  - `id`: Company unique identifier
  - `name`: Company name
  - `founded_at`: Founding date
  - `funding_total_usd`: Total funding raised
  - `status`: Current status (acquired/ipo/closed/operating)
  - `last_funding_at`: Date of last funding round
  - `country_code`: Country (USA, GBR, etc.)

**Example**:
```python
df = load_startups(filter_funded=True, min_funding=1_000_000)
print(f"Loaded {len(df)} companies")  # Loaded 10,011 companies
print(df['status'].value_counts())
```

#### load_temporal_data()
```python
from venture_scope.data.preprocessing import load_temporal_data

train, val, test = load_temporal_data(
    train_cutoff='2010-12-31',
    val_cutoff='2011-12-31',
    test_cutoff='2013-12-31'
)
```

**Parameters**:
- `train_cutoff` (str): End date for training set (default: '2010-12-31')
- `val_cutoff` (str): End date for validation set (default: '2011-12-31')
- `test_cutoff` (str): End date for test set (default: '2013-12-31')

**Returns**:
- Tuple of 3 DataFrames: (train, validation, test)
- Train: 7,008 companies (70%)
- Validation: 1,001 companies (10%)
- Test: 2,002 companies (20%)

**Example**:
```python
train, val, test = load_temporal_data()
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Train: 7008, Val: 1001, Test: 2002
```

### 3.2 Feature Engineering

#### calculate_features()
```python
from venture_scope.data.feature_engineering import calculate_features

features = calculate_features(
    company_df,
    cutoff_date='2010-12-31',
    include_composite=True
)
```

**Parameters**:
- `company_df` (DataFrame): Company data
- `cutoff_date` (str or datetime): Features calculated with info available by this date
- `include_composite` (bool, default=True): Include investment_score composite

**Returns**:
- `pandas.DataFrame` with 24 features:
  - **Funding** (6): funding_amount, funding_velocity, previous_rounds, days_since_founded, stage, funding_per_round
  - **Network** (5): investor_count, investor_quality_score, avg_investors_per_round, stage_encoded, has_top_tier_vc
  - **Operations** (8): capital_efficiency, burn_rate, runway_months, burn_multiple, traction_index, rule_of_40, estimated_revenue, monthly_burn
  - **Market** (5): sector, geography, stage_encoded, country_USA, sector_Tech

**Example**:
```python
features = calculate_features(train, cutoff_date='2010-12-31')
print(features.columns.tolist())
print(f"Feature importance order: {features.var().sort_values(ascending=False).index.tolist()[:5]}")
```

#### calculate_investment_score()
```python
from venture_scope.data.feature_engineering import calculate_investment_score

score = calculate_investment_score(
    company_data,
    weights={
        'rule_of_40': 0.25,
        'traction_index': 0.25,
        'capital_efficiency': 0.20,
        'burn_multiple': 0.15,
        'runway': 0.15
    }
)
```

**Returns**:
- `float` between 0-100: Composite investment score

**Example**:
```python
score = calculate_investment_score(company_data)
if score > 70:
    print("Strong investment candidate")
elif score > 50:
    print("Consider - mixed signals")
else:
    print("Weak signals")
```

---

## 4. Data Processing

### 4.1 Temporal Split

#### create_temporal_split()
```python
from venture_scope.data.preprocessing import create_temporal_split

train, val, test = create_temporal_split(
    df,
    train_end='2010-12-31',
    val_end='2011-12-31',
    test_end='2013-12-31',
    validate=True
)
```

**Parameters**:
- `df` (DataFrame): Full dataset
- `train_end` (str): Training set cutoff date
- `val_end` (str): Validation set cutoff date
- `test_end` (str): Test set cutoff date
- `validate` (bool, default=True): Run 18 automated validation tests

**Returns**:
- Tuple: (train_df, val_df, test_df)

**Validation Tests** (if `validate=True`):
- No temporal overlap between sets (4 tests)
- Feature cutoffs respected (6 tests)
- Target uses forward window (4 tests)
- No data leakage (4 tests)

**Example**:
```python
train, val, test = create_temporal_split(df, validate=True)
# Running 18 validation tests...
# PASSED: 17/18 tests (1 warning: partial investor data 5%)
```

---

## 5. Model Training & Prediction

### 5.1 Training

#### train_temporal_model()
```python
from venture_scope.ml.model_temporal import train_temporal_model

model, metrics = train_temporal_model(
    train_df,
    val_df,
    hyperparameters={
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 50,
        'min_samples_leaf': 20
    },
    optimize=True
)
```

**Parameters**:
- `train_df` (DataFrame): Training set (2000-2010)
- `val_df` (DataFrame): Validation set (2010-2011)
- `hyperparameters` (dict, optional): Model hyperparameters
- `optimize` (bool, default=True): Run grid search if True

**Returns**:
- `model`: Trained RandomForestClassifier
- `metrics`: Dictionary with validation metrics

**Example**:
```python
model, metrics = train_temporal_model(train, val, optimize=True)
print(f"Validation Recall: {metrics['recall']:.3f}")  # 0.932
print(f"Validation AUC: {metrics['auc_roc']:.3f}")    # 0.954
```

#### train_baseline_model()
```python
from venture_scope.ml.model import train_baseline_model

baseline_model = train_baseline_model(
    X_train,
    y_train,
    use_random_split=True
)
```

**Parameters**:
- `X_train` (DataFrame): Features
- `y_train` (Series): Target
- `use_random_split` (bool, default=True): Use random 80/20 split (creates look-ahead bias)

**Returns**:
- `RandomForestClassifier`: Trained baseline model

**Note**: This function intentionally uses random split to demonstrate look-ahead bias. Use `train_temporal_model()` for production.

### 5.2 Prediction

#### predict_startup()
```python
from venture_scope.ml.predict import predict_startup

result = predict_startup(
    funding_amount=15_000_000,
    investor_count=5,
    stage='Series A',
    sector='Software',
    country='USA',
    days_since_founded=730,
    capital_efficiency=0.35,
    model_path='data/models/random_forest_temporal.pkl'
)
```

**Parameters**:
- `funding_amount` (float): Total funding raised (USD)
- `investor_count` (int): Number of unique investors
- `stage` (str): Funding stage (Seed/Angel/Series A/Series B/Series C)
- `sector` (str): Primary sector (Tech/Healthcare/Fintech/etc.)
- `country` (str): Country code (USA/GBR/etc.)
- `days_since_founded` (int): Company age in days
- `capital_efficiency` (float): Revenue per funding dollar
- `model_path` (str, optional): Path to trained model

**Returns**:
- Dictionary:
```python
{
    'success_probability': 0.78,  # float, 0-1
    'investment_score': 67.2,     # float, 0-100
    'recommendation': 'CONSIDER', # str: STRONG/CONSIDER/CAUTIOUS
    'feature_contributions': {
        'funding_amount': 0.182,
        'investor_count': 0.134,
        'capital_efficiency': 0.098,
        # ... top 10 features
    },
    'predicted_class': 1,         # 0=failure, 1=success
    'confidence': 'medium'        # high/medium/low
}
```

**Example**:
```python
# Typical Series A SaaS startup
result = predict_startup(
    funding_amount=8_000_000,
    investor_count=4,
    stage='Series A',
    sector='Software',
    country='USA',
    days_since_founded=912,  # 2.5 years
    capital_efficiency=0.30
)

print(f"Success Probability: {result['success_probability']:.1%}")
print(f"Investment Score: {result['investment_score']:.1f}/100")
print(f"Recommendation: {result['recommendation']}")

# Output:
# Success Probability: 68.4%
# Investment Score: 62.3/100
# Recommendation: CONSIDER
```

#### batch_predict()
```python
from venture_scope.ml.predict import batch_predict

predictions = batch_predict(
    companies_df,
    model_path='data/models/random_forest_temporal.pkl'
)
```

**Parameters**:
- `companies_df` (DataFrame): Multiple companies with feature columns
- `model_path` (str): Path to trained model

**Returns**:
- `DataFrame` with added columns: `success_probability`, `predicted_class`, `investment_score`

**Example**:
```python
# Rank portfolio by predicted success
predictions = batch_predict(portfolio_df)
top_10 = predictions.nlargest(10, 'success_probability')
print(top_10[['name', 'success_probability', 'investment_score']])
```

---

## 6. Analysis Scripts

### 6.1 Model Comparison
```bash
python scripts/compare_models.py
```

**Outputs**:
- `results/figures/temporal_performance.png`
- `results/figures/feature_importance.png`
- `results/figures/calibration_curve.png`
- `results/figures/confusion_heatmap.png`
- `docs/MODEL_COMPARISON_REPORT.md`

**Programmatic Usage**:
```python
from venture_scope.analysis.compare_models import compare_models

comparison = compare_models(
    baseline_model,
    temporal_model,
    test_data
)

print(comparison['metrics_comparison'])
# {'baseline_recall': 0.901, 'temporal_recall': 0.938, 'improvement': 0.037}
```

### 6.2 Error Analysis
```bash
python scripts/error_analysis.py
```

**Outputs**:
- `results/figures/error_by_stage.png`
- `results/figures/error_by_funding.png`
- `results/figures/characteristics_comparison.png`
- `results/figures/probability_distributions.png`
- `docs/ERROR_ANALYSIS_REPORT.md`

**Programmatic Usage**:
```python
from venture_scope.analysis.error_analysis import analyze_errors

error_report = analyze_errors(
    model,
    test_data,
    segment_by=['stage', 'funding_bucket', 'sector']
)

print(error_report['stage_analysis'])
# {'Seed': {'miss_rate': 0.122, 'recall': 0.878}, ...}
```

### 6.3 Distribution Shift
```bash
python scripts/distribution_shift_analysis.py
```

**Outputs**:
- `results/figures/funding_shift_by_stage.png`
- `results/figures/market_context_changes.png`
- `results/figures/feature_distribution_shifts.png`
- `docs/DISTRIBUTION_SHIFT_REPORT.md`

**Programmatic Usage**:
```python
from venture_scope.analysis.distribution_shift import quantify_shift

shift_analysis = quantify_shift(
    train_2013=train_df,
    benchmarks_2025={
        'series_a_median': 12_000_000,
        'unicorn_count': 1200,
        # ... see DISTRIBUTION_SHIFT_REPORT.md
    }
)

print(shift_analysis['funding_multipliers'])
# {'Seed': 3.0, 'Series A': 2.4, 'Series B': 2.0, ...}
```

---

## 7. Testing

### 7.1 Run All Tests
```bash
# Run full test suite
pytest tests/ -v

# Expected output:
# test_temporal_validation.py::test_no_temporal_overlap PASSED
# test_temporal_validation.py::test_chronological_ordering PASSED
# ... (18 tests total)
# ==================== 17 passed, 1 warning in 12.34s ====================
```

### 7.2 Run Specific Test Categories
```bash
# Temporal integrity tests only
pytest tests/test_temporal_validation.py -v

# Feature engineering tests
pytest tests/test_feature_engineering.py -v

# Model tests
pytest tests/test_model.py -v
```

### 7.3 Programmatic Testing
```python
from venture_scope.validation.temporal_tests import run_all_tests

results = run_all_tests(train, val, test)
print(f"Tests passed: {results['passed']}/{results['total']}")
# Tests passed: 17/18

if results['warnings']:
    print(f"Warnings: {results['warnings']}")
    # Warnings: ['Partial investor data missing (5%)']
```

---

## 8. CLI Tools

### 8.1 Interactive Prediction
```bash
python src/venture_scope/ml/predict.py
```

**Interactive Prompts**:
```
=== VENTURE-SCOPE Startup Predictor ===

Enter funding amount (USD): 10000000
Enter investor count: 5
Enter stage (Seed/Angel/Series A/B/C): Series A
Enter sector (Software/Healthcare/Fintech/etc.): Software
Enter country (USA/GBR/etc.): USA
Enter days since founded: 1095

--- PREDICTION RESULTS ---
Success Probability: 72.3%
Investment Score: 65.8/100
Recommendation: CONSIDER

Top Contributing Features:
  1. funding_amount: 18.2%
  2. investor_count: 13.4%
  3. capital_efficiency: 9.8%
  ...

Predict another? (y/n):
```

### 8.2 Model Training CLI
```bash
python src/venture_scope/ml/model_temporal.py --optimize --save
```

**Options**:
- `--optimize`: Run hyperparameter grid search
- `--save`: Save trained model to data/models/
- `--evaluate`: Evaluate on test set after training

### 8.3 Batch Analysis
```bash
python scripts/batch_analysis.py --input portfolio.csv --output predictions.csv
```

**Input CSV Format**:
```csv
name,funding_amount,investor_count,stage,sector,country,days_since_founded
Startup A,5000000,3,Series A,Software,USA,730
Startup B,15000000,8,Series B,Fintech,GBR,1460
...
```

**Output**: Same CSV with added columns: `success_probability`, `investment_score`, `recommendation`

---

## 9. File Structure

### 9.1 Data Files
```
data/
├── raw/
│   └── crunchbase_companies.csv          # Original Crunchbase data
├── processed/
│   ├── train_2000_2010.csv              # Training set (7,008 companies)
│   ├── val_2010_2011.csv                # Validation set (1,001 companies)
│   └── test_2011_2013.csv               # Test set (2,002 companies)
└── models/
    ├── random_forest.pkl                 # Baseline model (random split)
    └── random_forest_temporal.pkl        # Final model (temporal split)
```

### 9.2 Result Files
```
results/
├── figures/
│   ├── temporal_performance.png          # Recall progression
│   ├── feature_importance.png            # Top 10 features
│   ├── calibration_curve.png             # Probability calibration
│   ├── confusion_heatmap.png             # Confusion matrix
│   ├── error_by_stage.png                # Miss rates by stage
│   ├── error_by_funding.png              # Miss rates by funding
│   ├── characteristics_comparison.png     # FN vs TP comparison
│   ├── probability_distributions.png      # TN/FP/FN/TP probabilities
│   ├── funding_shift_by_stage.png        # 2013 vs 2025 funding
│   ├── market_context_changes.png        # Unicorns, exits, VC deployed
│   └── feature_distribution_shifts.png   # Feature distributions 2013 vs 2025
└── model_performance.txt                  # Logs and metrics
```

### 9.3 Documentation Files
```
docs/
├── README.md                              # Project overview
├── EXECUTIVE_SUMMARY.md                   # 1-page summary
├── LITERATURE_REVIEW.md                   # Academic positioning
├── METHODOLOGY.md                         # Technical details
├── MODEL_COMPARISON_REPORT.md             # Baseline vs Temporal
├── ERROR_ANALYSIS_REPORT.md               # Segment analysis
└── DISTRIBUTION_SHIFT_REPORT.md           # 2013 vs 2025
```

---

## 10. API Reference Summary

### Data Loading
- `load_startups()` - Load raw Crunchbase data
- `load_temporal_data()` - Load temporal splits

### Feature Engineering
- `calculate_features()` - Compute 24 features with cutoff dates
- `calculate_investment_score()` - Composite score (0-100)

### Model Training
- `train_temporal_model()` - Train with temporal validation
- `train_baseline_model()` - Train with random split (for comparison)

### Prediction
- `predict_startup()` - Single company prediction
- `batch_predict()` - Multiple companies

### Analysis
- `compare_models()` - Baseline vs Temporal comparison
- `analyze_errors()` - Segment-specific error analysis
- `quantify_shift()` - Distribution shift 2013 vs 2025

### Validation
- `run_all_tests()` - 18 temporal integrity tests

---

## 11. Performance Characteristics

### Computational
- Full pipeline: 15 minutes (MacBook Pro M1)
- Temporal split: 2-3 minutes
- Model training: 5 minutes (300 trees, 7,008 samples)
- Feature engineering: 3 minutes (10,011 companies, 24 features)
- Memory: 2 GB peak

### Model Performance
- **Test Set (2011-2013)**: 2,002 companies
- **Recall**: 93.8% (captured 2182/2326 successes)
- **Precision**: 75.9%
- **F1-Score**: 83.7%
- **AUC-ROC**: 0.956
- **False Negative Rate**: 6.2% (missed 144 winners)

### Comparison
- Temporal Model: 93.8% recall
- Baseline Model: 90.1% recall
- Improvement: +3.7 percentage points

---

## 12. Common Usage Patterns

### Pattern 1: Full Pipeline
```python
from venture_scope.data.preprocessing import load_temporal_data
from venture_scope.ml.model_temporal import train_temporal_model
from venture_scope.ml.evaluation import evaluate_model

# Load data
train, val, test = load_temporal_data()

# Train model
model, val_metrics = train_temporal_model(train, val, optimize=True)

# Evaluate
test_metrics = evaluate_model(model, test)
print(f"Test Recall: {test_metrics['recall']:.3f}")  # 0.938
```

### Pattern 2: Portfolio Screening
```python
from venture_scope.ml.predict import batch_predict

# Load portfolio
portfolio = load_portfolio('portfolio.csv')

# Predict all
predictions = batch_predict(portfolio, model_path='data/models/random_forest_temporal.pkl')

# Filter top candidates
top_candidates = predictions[predictions['success_probability'] > 0.70]
top_candidates = top_candidates.sort_values('investment_score', ascending=False)

print(f"Top 10 Investment Candidates:")
print(top_candidates.head(10)[['name', 'success_probability', 'investment_score']])
```

### Pattern 3: Feature Importance Analysis
```python
from venture_scope.ml.model_temporal import get_feature_importance

importance = get_feature_importance(model, feature_names)

# Top 5 features
top5 = importance.nlargest(5, 'importance')
print(top5)
# funding_amount: 18.2%
# investor_count: 13.4%
# capital_efficiency: 9.8%
# days_since_founded: 8.7%
# previous_rounds: 7.6%
```

---

## 13. Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'venture_scope'"

**Solution**:
```bash
pip install -e .
```

### Issue: "18 tests failed - temporal overlap detected"

**Solution**: Check that temporal split is correctly implemented:
```python
train, val, test = create_temporal_split(df, validate=True)
# Review error messages for specific overlap
```

### Issue: "Model prediction gives same probability for all inputs"

**Solution**: Check feature scaling and model loading:
```python
# Verify model loaded correctly
import pickle
with open('data/models/random_forest_temporal.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.n_estimators)  # Should be 300
```

### Issue: "ImportError: cannot import name 'calculate_features'"

**Solution**: Check Python path and reinstall:
```bash
pip uninstall venture-scope
pip install -e .
```

---

## 14. Version History

**v2.0 (January 2025)** - Temporal Validation Release
- Added temporal validation framework
- 18 automated tests for temporal integrity
- 93.8% recall on test set
- Error analysis by segment
- Distribution shift quantification

**v1.0 (December 2024)** - Initial Release
- Basic random split implementation
- 90.1% recall baseline
- 7 engineered features

---

## 15. Contact & Support

**Author**: Arthur Pillet  
**Email**: arthur.pillet@unil.ch  
**Institution**: HEC Lausanne, University of Lausanne

**Documentation**: See `/docs/` folder for comprehensive guides

**Issues**: Please document any bugs or feature requests

---

**End of API Documentation**
```
