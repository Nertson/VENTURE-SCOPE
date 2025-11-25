# API Documentation

## Installation
```bash
pip install -e .
```

## Core Functions

### load_startups()
```python
from venture_scope.ingest.loaders import load_startups
df = load_startups(filter_funded=True)
```
**Parameters:**
- `filter_funded` (bool): Keep only funded companies

**Returns:**
- DataFrame with 7 columns

**Example:**
```python
df = load_startups()
print(df.head())
```

### predict_startup()
```python
from venture_scope.ml.predict import predict_startup

result = predict_startup(
    funding_amount=15_000_000,
    stage='Series A',
    sector='software',
    country='USA'
)
```

**Returns:**
```python
{
    'success_probability': 0.78,
    'investment_score': 67.2,
    'feature_contributions': {...}
}
```

## Models

### RandomForestModel
- Path: `results/models/random_forest.pkl`
- Features: 116
- Performance: 76% accuracy, 90% recall

## CLI Tools

### Interactive Prediction
```bash
python src/venture_scope/ml/predict.py
```

### Model Comparison
```bash
python src/venture_scope/ml/model_comparison.py
```
