# Examples - Usage Guide

## 🎯 What's in this folder?

- `missing_data_analysis.py` - Statistical analysis
- `create_visualizations.py` - Generate figures

## 🚀 Quick Start

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

✅ Success Probability: 78.3%
📊 Investment Score: 67.2/100
```

## 2. Generate Visualizations
```bash
python examples/create_visualizations.py
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

**FULL DEMO**
```bash
# =====================================================
# 🎯 VENTURE-SCOPE : DÉMONSTRATION COMPLÈTE
# =====================================================

cd /files/VENTURE-SCOPE

echo "======================================================================"
echo "🚀 VENTURE-SCOPE DEMO - VC Investment Decision Support System"
echo "======================================================================"
echo ""

# -------------------- 1. STRUCTURE --------------------
echo "📂 STEP 1: Project Structure"
echo "----------------------------------------------------------------------"
tree -L 2 -I '__pycache__|.git|*.pyc|.pytest_cache' || ls -R
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 2. TESTS --------------------
echo "🧪 STEP 2: Running Tests"
echo "----------------------------------------------------------------------"
pytest tests/ -v
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 3. DATA OVERVIEW --------------------
echo "📊 STEP 3: Data Overview"
echo "----------------------------------------------------------------------"
python -c "
import pandas as pd
df = pd.read_csv('data/processed/startups_scored.csv')
print(f'Total Companies: {len(df):,}')
print(f'Columns: {len(df.columns)}')
print(f'\nSample:')
print(df[['company', 'stage', 'sector', 'funding_amount', 'investment_score']].head(10))
print(f'\nInvestment Score Distribution:')
print(df['investment_score'].describe())
"
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 4. MODEL COMPARISON --------------------
echo "🔬 STEP 4: Model Comparison (4 Algorithms)"
echo "----------------------------------------------------------------------"
echo "Comparing: Logistic Regression, Random Forest, Gradient Boosting, SVM"
echo ""
python src/venture_scope/ml/model_comparison.py
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 5. TOP RECOMMENDATIONS --------------------
echo "🏆 STEP 5: Top 10 Investment Recommendations"
echo "----------------------------------------------------------------------"
python -c "
import pandas as pd
df = pd.read_csv('results/top_100_startups.csv')
print(df[['company', 'stage', 'sector', 'investment_score', 'funding_amount']].head(10).to_string(index=False))
"
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 6. MISSING DATA ANALYSIS --------------------
echo "📈 STEP 6: Missing Data Statistical Analysis"
echo "----------------------------------------------------------------------"
echo "Question: Do small firms report less data?"
echo ""
python examples/missing_data_analysis.py
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 7. VISUALIZATIONS --------------------
echo "🎨 STEP 7: Generating Professional Visualizations"
echo "----------------------------------------------------------------------"
python examples/create_visualizations.py
echo ""
echo "📁 Visualizations created in: results/figures/"
ls -lh results/figures/*.png
echo ""
read -p "Press ENTER to continue..."
echo ""

# -------------------- 8. INTERACTIVE PREDICTION --------------------
echo "🔮 STEP 8: Interactive Prediction (LIVE DEMO)"
echo "----------------------------------------------------------------------"
echo "Let's predict success for a sample startup!"
echo ""
python src/venture_scope/ml/predict.py
echo ""

# -------------------- 9. SUMMARY --------------------
echo ""
echo "======================================================================"
echo "✅ DEMO COMPLETE!"
echo "======================================================================"
echo ""
echo "📊 What we demonstrated:"
echo "  1. ✅ Complete project structure"
echo "  2. ✅ Automated tests (all passing)"
echo "  3. ✅ Dataset overview (27,874 startups)"
echo "  4. ✅ Formal model comparison (4 algorithms)"
echo "  5. ✅ Top 100 investment recommendations"
echo "  6. ✅ Missing data statistical analysis"
echo "  7. ✅ Professional visualizations"
echo "  8. ✅ Live prediction demo"
echo ""
echo "📈 Key Results:"
echo "  • Random Forest selected (76% accuracy, 90% recall)"
echo "  • Missing data: 2.86x funding difference (p < 0.001)"
echo "  • Top feature: funding_amount (25.9% importance)"
echo ""
echo "📁 All results saved in: results/"
echo "======================================================================"
```
