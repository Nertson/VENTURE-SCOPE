# Examples

## 1. Interactive Startup Prediction
```bash
python src/venture_scope/ml/predict.py
```

Interactively predict success probability for a startup.

**Example**:
```
Enter funding amount: 15000000
Enter stage: Series A
Enter sector: software
Enter country: USA
Enter investors count: 8
Enter founded year: 2020

✅ Success Probability: 78.3%
📊 Investment Score: 67.2/100
```

## 2. Generate Visualizations
```bash
python examples/create_visualizations.py
```

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

Performs rigorous statistical analysis:
- T-test: funding comparison (p < 0.001)
- Chi-square: success rate association
- Stage/sector breakdown

Answers professor's question: "Do small firms report less?"

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
