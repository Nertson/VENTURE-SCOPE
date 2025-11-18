# API Documentation

## Interactive Predictor

### Usage
```python
from venture_scope.ml.predict import predict_startup

result = predict_startup(
    funding_amount=15_000_000,
    stage='Series A',
    sector='software',
    country='USA',
    investors_count=8,
    founded_year=2020
)

print(f"Success Probability: {result['success_probability']:.1%}")
print(f"Investment Score: {result['investment_score']:.1f}/100")
```

### Command Line
```bash
python src/venture_scope/ml/predict.py
```

Follow interactive prompts to get predictions.

## Model Comparison
```bash
python src/venture_scope/ml/model_comparison.py
```

Runs formal comparison of 4 ML algorithms.

## Analysis Scripts

### Missing Data Analysis
```bash
python examples/missing_data_analysis.py
```

### Visualizations
```bash
python examples/create_visualizations.py
```

Generates 5 professional figures in `results/figures/`.
