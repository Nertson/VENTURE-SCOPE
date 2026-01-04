# Examples - Usage Guide

## What's in this folder?

- `missing_data_analysis.py` - Statistical analysis
- `create_visualizations_v2.py` - Generate figures

## Quick Start

### 1. Interactive Prediction
```bash
python src/venture_scope/ml/predict.py
```

**What it does:**
- Asks for startup info (funding, stage, sector...)
- Predicts success probability
- Shows investment score

**Example session:**
```
Enter funding amount: 15000000
Enter stage: Series A
Enter sector: software

Success Probability: 78.3%
Investment Score: 67.2/100
```

## 2. Generate Visualizations
```bash
python examples/create_visualizations_v2.py
```
**What it does:**
- Creates 5 professional charts
- Saves to `results/figures/`

**Output:**

Creates 5 professional visualizations:
- `model_comparison.png` - 4 models compared
- `confusion_matrix.png` - Random Forest performance
- `feature_importance.png` - Top 10 features
- `missing_data_analysis.png` - Funding comparison
- `roc_curves.png` - ROC curves

## 3. Missing Data Statistical Analysis
```bash
python examples/missing_data_analysis.py
```

**What it does:**
- T-test on funding amounts
- Chi-square on success rates
- Answers: "Do small firms report less?"

**Output:**
- Console report with statistics
- `results/missing_data_analysis.csv`

## 4. Model Comparison
```bash
python src/venture_scope/ml/model_comparison.py
```

Formal comparison of 4 algorithms:
- Logistic Regression
- Random Forest (selected)
- Gradient Boosting
- SVM

Results saved to `results/model_comparison.csv`.

## 5. Run Tests
```bash
pytest tests/ -v
```

Validates:
- Model exists
- Data integrity
- Results completeness
- Visualizations generated


## 6. TOP 10 investments recommendations
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/top_100_startups.csv')
print(df[['company', 'stage', 'investment_score']].head(10))
"

```


