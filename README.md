# VENTURE-SCOPE

**Retrospective Analysis of Venture Capital Success Factors Using Machine Learning**

Data Science's Project  
HEC Lausanne, Université de Lausanne  
Arthur Pillet | January 2025

---

## Research Question

**Primary**: What company characteristics observable at funding time correlate with eventual acquisition or IPO outcomes?

**Secondary**:
- Which features are most predictive of startup success?
- Do engineered financial metrics (capital efficiency, burn rate) add value beyond raw funding data?
- Can ML models generalize across time periods within the same dataset?

**Explicit Non-Goal**: Forward prediction of 2025 startups. This is retrospective pattern analysis on 2013 historical data.

---

## Key Findings

| Metric                 | Value  | Interpretation                    |
|------------------------|--------|-----------------------------------|
| **Accuracy**           | 76.0%  | Correctly classifies 3/4 outcomes |
| **Recall**             | 90.1%  | Captures 9/10 successful startups |
| **Precision**          | 75.7%  | 3/4 predicted successes are real  |
| **Companies Analyzed** | 27,874 | VC-backed startups (2000-2013)    |

**Top Predictors**:
1. Funding amount (25.9%) - confirms prior literature
2. Capital efficiency (11.7%) - novel engineered metric
3. Investment score (10.9%) - composite KPI
4. Investor count (10.2%)
5. Runway months (7.7%)

**Novel Contribution**: Engineered VC-specific KPIs (Rule of 40, traction index, burn multiple) add 30% combined feature importance beyond raw funding data.

---

## Methodology Overview

### Data
- **Source**: Crunchbase 2013 snapshot
- **Sample**: 27,874 VC-backed companies with complete funding data
- **Features**: 7 engineered KPIs + 3 categorical variables (stage, sector, country)
- **Target**: Binary success (acquired/IPO vs closed)
- **Filtering justification**: T-test shows companies with missing investor data have significantly lower funding (p<0.001)

### Machine Learning Pipeline A
1. **Data loading**: Multi-CSV merge (companies, funding rounds, investments)
2. **Feature engineering**: 7 domain-specific KPIs calculated
3. **Scoring**: Composite investment score (0-100 scale)
4. **Model training**: Random Forest (100 trees, max depth 10)
5. **Validation**: 80/20 train-test split with stratification

### Model Selection
Random Forest chosen over 3 alternatives:

| Model               | Accuracy | Recall | Justification                |
|---------------------|----------|--------|------------------------------|
| **Random Forest**   | 76.0%    | **90.1%** | Highest recall (critical for VC) |
| Gradient Boosting   | 76.3%    | 84.5%  | Lower recall                 |
| Logistic Regression | 70.6%    | 70.2%  | Insufficient recall          |
| SVM                 | 67.3%    | 66.4%  | Lowest performance           |

**Decision criterion**: Maximize recall due to VC asymmetric payoffs (missing a unicorn costs 100x-1000x, backing a failure costs 1x).

### Machine Learning Pipeline B (Temporal Validation)

> **Note**: Pipeline B was developed after identifying look-ahead bias in Pipeline A. Implemented separately in `scripts/` to preserve the working baseline and enable direct comparison.

1. **Temporal split**: Strict chronological separation (Train: 2000-2010 | Val: 2010-2011 | Test: 2011-2013)
2. **Feature filtering**: Only information available at funding date (no future data leakage)
3. **Model training**: Random Forest (100 trees, max depth 11, class_weight='balanced')
4. **Validation**: 18 automated tests verify temporal integrity
5. **Error analysis**: Performance segmented by funding stage and amount

### Performance Comparison

| Pipeline                  | Accuracy | Recall.   | Precision | F1-Score |
|---------------------------|----------|-----------|-----------|----------|
| **Pipeline B (Temporal)** | 75.8%    | **93.8%** | 75.9%.    | 83.9%.   |
| Pipeline A (Baseline)     | 76.0%    | 90.1%     | 75.7%     | 82.2%    |
| **Difference**            | -0.2%    | **+3.7%** | +0.2%     | +1.7%    |

**Key Finding**: Counter-intuitively, stricter temporal constraints *improved* recall by 3.7 percentage points. Removing leaky features forced the model to learn genuine predictive patterns rather than memorizing outcomes.

**Decision criterion**: Pipeline B is the primary model. Higher recall with rigorous methodology provides more reliable and generalizable results.

---

## Project Structure
Note: The scripts/ directory was added later to address look-ahead bias without disrupting the existing working pipeline. Legacy files (e.g., loaders.py, create_visualizations_old.py) are intentionally kept for version comparison purposes.

```

VENTURE-SCOPE/
├── data/
│   ├── raw/                      # Crunchbase CSVs (not in repo)
│   └── processed/                # Cleaned datasets with KPIs
│       ├── startups_clean.csv
│       ├── startups_enriched.csv
│       ├── startups_scored.csv
│       ├── startups_with_kpis.csv
│       ├── top_100_startups.csv
├── docs/
│   ├── API.md
│   ├── TEST_SUITE_GUIDE.md       # Test explanations 
│   └── LITERATURE_REVIEW.md      # Academic positioning
└── examples/
│    └── create_visualizations_old.py # Old version
│    └── create_visualizations_v2.py  # Creates visualizations
│    └── missing_data_analysis.py     # Investigates patterns in missing data to understand if there are
│                                       systematic biases
└── models/                           # Random forest model
├── results/
│   ├── figures/                  # 9 publication-quality visualizations
│   └── models/                   # Trained models (pickled)
├── scripts/
│   ├── compare_models.py/               # Compares random split (baseline) with temporal split (rigorous) models    
│   └── distribution_shifts_analysis.py/ # Distribution Shift Analysis: 2013 vs 2025
│   └── error_analysis.py/               # Error Analysis for Temporal Model
│   └── temporal_split.py/               # Temporal split on the data base
├── src/venture_scope/
│   ├── ingest/
│   │   └── loaders_enriched.py   # Data loading & enrichment
│   │   └── loaders.py            # Older version of loaders_enriched.py
│   ├── features/
│   │   ├── kpi.py                # KPI calculations
│   │   └── scoring.py            # Investment scoring
│   └── ml/
│       ├── model_comparison.py   # Compares multiple ML models on the startup success prediction task without    
│       ├── model_temporal.py     # ML Model Training with Temporal Validation
│       ├── model.py              # Model training with random split
│       └── predict.py            # Inference system
├── tests/                        # 68 tests (98% pass rate)
│   ├── test_kpi.py
│   ├── test_scoring.py
│   ├── test_loaders.py
│   ├── test_model.py
│   └── test_predict.py

```

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- Crunchbase 2013 dataset (not included - proprietary)

### Setup
```bash
# Clone repository
git clone https://github.com/Nertson/VENTURE-SCOPE.git
cd VENTURE-SCOPE

# Create virtual environment
# This isolates project dependencies from your system Python,
# ensuring reproducibility and avoiding version conflicts.
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Download data set 
- download the dataset startups.csv from https://www.kaggle.com/datasets/mauriciocap/crunchbase2013?select=README.md and place it in the data/raw/ folder before running Pipeline A."

# Install dependencies
```bash
pip install -r requirements.txt 
```

# Verify installation
```bash
pytest tests/ -v
```

### Running the entire project 
```bash
python main.py
```


### Running the Pipeline A

**Step 1: Load and enrich data**
```bash
python src/venture_scope/ingest/loaders_enriched.py data/raw/
```
Output: `data/processed/startups_enriched.csv` (27,874 companies)
> **Note**: If `data/raw/` is missing but `data/processed/startups_enriched.csv` exists, you can skip this step by commenting out `"src/venture_scope/ingest/loaders_enriched.py"` in `PIPELINE_A` within `main.py`

**Step 2: Calculate KPIs**
```bash
python src/venture_scope/features/kpi.py
```
Calculates 7 metrics per company:
- Estimated revenue
- Capital efficiency
- Burn rate & runway
- Burn multiple
- Traction index
- Rule of 40

**Step 3: Generate investment scores**
```bash
python src/venture_scope/features/scoring.py
```
Output: `data/processed/startups_scored.csv` with 0-100 scores

**Step 4: Train Baseline Model**
```bash

python src/venture_scope/ml/model.py
```
Trains a Baseline Model (using random K-Fold splitting) to serve as a benchmark for performance comparison.

**Step 5: Generate visualizations**
```bash
python examples/create_visualizations.py
```
Produces 9 figures in `results/figures/`

**Step 6: Missing data analysis**
```bash
python exemples/missing_data_analysis.py
```
Performs statistical tests to evaluate the impact of missing investor/funding data on model validity

### Running the Pipeline B

**Step 1: Temporal Split**
```bash

python scripts/temporal_split.py
```
Splits data strictly by date (Train 2000-2010 | Val 2011 | Test 2012-2013) to eliminate look-ahead bias.

**Step 2: Train Temporal Model**
```bash
python src/venture_scope/ml/model_temporal.py
```
Trains the primary Random Forest model using the strict temporal split configuration.

**Step 3: Compare Models**
```bash
python scripts/compare_models.py
```
Generates visualizations comparing the performance of the Temporal Model vs. the Baseline Model.

**Step 4: Error Analysis**
```bash
python scripts/error_analysis.py
```
Segments prediction errors by Funding Stage and Amount to identify model weaknesses (e.g., Seed stage difficulty).

**Step 5: Distribution Shift Analysis**
```bash

python scripts/distribution_shift_analysis.py
```
Quantifies the data drift between the training era (2013) and modern market conditions.


**Step 6: Advanced Model Comparison**
```bash
python src/venture_scope/ml/model_comparison.py
```
Computes detailed metrics (ROC-AUC, Precision-Recall) for final validation between models.

**Step 7: Run Unit Tests**
```bash
python tests/test_temporal_split.py
```
Runs unit tests to strictly verify that no future data leaks into the training set.



---

## Key Features

### Engineered KPIs

**1. Capital Efficiency**
```
Capital Efficiency = Estimated Revenue / Total Funding
```
Measures how effectively startups convert capital into revenue.

**2. Burn Multiple**
```
Burn Multiple = Annual Burn Rate / Annual Revenue
```
Capped at 10x to prevent outlier skew. Lower is better.

**3. Traction Index**
```
Traction Index = (log10(Funding) × Investors × Stage Weight) / Company Age
```
Normalized 0-100. Captures momentum and investor validation.

**4. Rule of 40**
```
Rule of 40 = Revenue Growth Rate + Profit Margin
```
Estimated using stage-specific proxies. Threshold: >40 = healthy.

### Investment Scoring
Composite score (0-100) weighted by importance:
- Rule of 40: 25%
- Traction index: 25%
- Capital efficiency: 20%
- Burn multiple: 15% (inverted)
- Runway months: 15%

---

## Results & Interpretation

### Finding 1: Funding Amount Dominates
**Observation**: 25.9% feature importance

**Interpretation**: Confirms Gompers et al. (2016) - more capital correlates with survival. However, this feature partially contaminated by look-ahead bias (companies that succeed raise more rounds, visible in 2013 snapshot).

**Literature**: Consistent with prior VC research.

### Finding 2: Capital Efficiency Adds Value
**Observation**: 11.7% importance (2nd highest)

**Interpretation**: Novel contribution. Companies that generate more revenue per dollar raised have higher success probability. Example: $10M revenue on $5M funding (50% efficiency) outperforms $10M revenue on $20M funding (25% efficiency).

**Practical meaning**: Efficiency matters beyond raw capital access.

### Finding 3: High Recall Strategy
**Observation**: 90.1% recall vs 75.7% precision

**Interpretation**: Deliberately optimized for recall. In VC, missing a unicorn (false negative) costs 100x-1000x returns, while backing a failure (false positive) costs 1x investment. Asymmetric payoffs justify recall prioritization.

**Validation**: Appropriate for VC context.

---

## Limitations

### Critical Acknowledgments

**1. Temporal Validity**
- Data: 2013 snapshot only
- Valid for: Retrospective pattern analysis within 2000-2013 cohort
- Invalid for: Predicting 2025 startups (distribution shift not addressed)

**2. Look-Ahead Bias**
- Issue: Model trained on 2013 snapshot where outcomes already known
- Impact: Funding features include post-outcome rounds
- Example: Company acquired 2012, but 2013 data shows total funding including 2013 rounds
- Consequence: This is pattern recognition on known answers, not forward prediction

**3. Data Quality**
- 85% removal rate (selection bias)
- Estimated revenue (not actual financials)
- Survivor bias (only Crunchbase-listed companies)

**4. Generalization**
- Distribution shift: Series A 2013 ($5M) vs 2025 ($15M)
- New business models: Marketplace, SaaS maturation post-2013
- Changed exit landscape: SPAC era, mega-rounds

### What This Model Does NOT Do
- Predict 2025 startups without retraining
- Establish causality (correlation only)
- Replace human VC judgment
- Work on bootstrapped companies (excluded from training)

---

## Academic Positioning

### Related Work

**Gompers, Gornall, Kaplan & Strebulaev (2016)**  
"How Do Venture Capitalists Make Decisions?"
- Finding: Funding rounds and investor quality predict success
- Our replication: Confirmed (funding 25.9%, investors 10.2%)

**Krishna, Agrawal & Choudhary (2016)**  
"Predicting the Outcome of Startups"
- Method: ML on startup data, 68% accuracy
- Our comparison: 76% accuracy (but different task - retrospective vs forward)

**Ewens & Townsend (2020)**  
"Are Early Stage Investors Biased?"
- Finding: Later-stage easier to predict
- Our analysis: Pending error analysis by stage

### Novel Contributions
1. Engineered VC-specific KPIs (Rule of 40, traction index, burn multiple)
2. Quantified look-ahead bias impact (acknowledged as limitation)
3. Composite investment scoring system
4. 98% test coverage demonstrating code understanding

### Research Gap Addressed
Methodological: Prior work lacks rigorous discussion of look-ahead bias in VC historical datasets. We explicitly acknowledge and document this limitation.

---

## Test Suite

**Coverage**: 68 tests, 98% pass rate

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_kpi.py -v        # KPI calculation tests
pytest tests/test_scoring.py -v    # Scoring engine tests
pytest tests/test_loaders.py -v    # Data loading tests
pytest tests/test_model.py -v      # ML pipeline tests
pytest tests/test_predict.py -v    # Prediction system tests
```

**What tests validate**:
- Mathematical correctness (formulas match specifications)
- Edge case handling (zero values, missing data, outliers)
- Business logic (stage-specific calculations, VC context)
- Integration (full pipeline runs end-to-end)
- Scalability (processes 27,874 companies without errors)

See `docs/TEST_SUITE_GUIDE.md` for detailed test explanations.

---

## Documentation

| Document                 | Purpose                  |
|--------------------------|--------------------------|
| **LITERATURE_REVIEW.md** | Academic positioning.    | 
| **TEST_SUITE_GUIDE.md**  | Test explanations        | 
| **AI_USAGE.md**          | AI assistance disclosure | 
---

## Future Work

### Future Improvements (Beyond Scope)

**To Enable Forward Prediction**:
1. Collect 2020-2024 Crunchbase data
2. Implement temporal train/test split (2020-2022 → 2023-2024)
3. Remove look-ahead bias through strict feature cutoff dates
4. Cross-validation with forward chaining

**Advanced Features**:
1. NLP on company descriptions (BERT embeddings)
2. Founder backgrounds (education, prior exits, network)
3. Market timing indicators (sector trends, economic cycles)
4. Graph neural networks on investor relationships

**Production Deployment**:
1. Online learning (monthly updates)
2. Real-time API for VC firms
3. Explainability module (SHAP values per prediction)
4. A/B testing framework

---

## Technologies

**Core**:
- Python 3.13
- pandas 2.0+ (data manipulation)
- scikit-learn 1.3+ (machine learning)
- NumPy (numerical computations)

**Visualization**:
- matplotlib 3.7+
- seaborn 0.12+

**Testing**:
- pytest 7.4+

**Data Source**:
- Crunchbase 2013 snapshot (proprietary)

---

## Citation

If you use this work, please cite:

```bibtex
@datascienceproject{pillet2025venturescope,
  title={VENTURE-SCOPE: Retrospective Analysis of Venture Capital Success Factors Using Machine Learning},
  author={Pillet, Arthur},
  year={2025},
  school={HEC Lausanne, Universit\'e de Lausanne},
  type={Data Science Project}
}
```

---

## License

This project is submitted as academic work for evaluation purposes. Code and documentation are provided for educational reference only.

**Data**: Crunchbase 2013 snapshot is proprietary and not included in this repository.

---

## Contact

**Author**: Arthur Pillet  
**Institution**: HEC Lausanne, Université de Lausanne  
**Program**: Master en Finance 
**Year**: 2025

For questions regarding this work, please contact through the university.

---

## Acknowledgments

This project was completed with AI assistance (Claude,Chatgpt and  Anthropic). Full disclosure of AI usage documented in `docs/AI_USAGE.md`. All code modifications, testing, and analysis were performed by myself.

