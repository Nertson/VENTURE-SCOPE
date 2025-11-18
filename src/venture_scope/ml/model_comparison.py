"""
Formal Model Comparison for VENTURE-SCOPE

Compares multiple ML models on the startup success prediction task.
Addresses professor feedback: "include a prominent section on the 
application of data science / machine learning methods, including 
if necessary a formal model comparison procedure."
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import time

print("=" * 70)
print("🔬 VENTURE-SCOPE: Formal Model Comparison")
print("=" * 70)


# ==================== DATA LOADING ====================

def load_data():
    """Load processed data with KPIs."""
    data_path = Path("data/processed/startups_scored.csv")
    
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("   Run: python src/venture_scope/analysis/kpi_calculator.py")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} companies with KPIs")
    return df


# ==================== FEATURE PREPARATION ====================

def prepare_ml_data(df):
    """
    Prepare features and labels for ML models.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Filter to known outcomes
    df_ml = df[df['status'].isin(['acquired', 'ipo', 'closed'])].copy()
    
    # Create binary label
    df_ml['success'] = (df_ml['status'].isin(['acquired', 'ipo'])).astype(int)
    
    print(f"\n📊 ML Dataset:")
    print(f"   Total: {len(df_ml):,} companies")
    print(f"   Success: {df_ml['success'].sum():,} ({df_ml['success'].mean()*100:.1f}%)")
    print(f"   Failure: {(1-df_ml['success']).sum():,} ({(1-df_ml['success'].mean())*100:.1f}%)")
    
    # Select features
    numeric_features = [
        'funding_amount', 'investors_count', 'rule_of_40', 
        'traction_index', 'capital_efficiency', 'burn_multiple',
        'runway_months', 'investment_score'
    ]
    
    categorical_features = ['stage', 'sector', 'country']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(
        df_ml[numeric_features + categorical_features], 
        columns=categorical_features,
        drop_first=False
    )
    
    X = df_encoded
    y = df_ml['success']

    # Handle missing values (fill NaN with 0 for numeric, mean for others)
    print(f"\n🧹 Cleaning data...")
    print(f"   NaN values before: {X.isna().sum().sum()}")

    # Fill numeric NaN with 0
    X = X.fillna(0)

    print(f"   NaN values after: {X.isna().sum().sum()}")
    print(f"   ✅ Data cleaned!")
    
    # Train/test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n✅ Features prepared: {X.shape[1]} features")
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ==================== MODEL DEFINITIONS ====================

def get_models():
    """
    Define models to compare.
    
    Returns:
        Dictionary of {model_name: (model, needs_scaling)}
    """
    models = {
        'Logistic Regression': (
            LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            ),
            True  # Needs feature scaling
        ),
        
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            False  # Tree-based, no scaling needed
        ),
        
        'Gradient Boosting': (
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            False  # Tree-based, no scaling needed
        ),
        
        'Support Vector Machine': (
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True,  # Enable probability estimates
                class_weight='balanced'
            ),
            True  # Needs feature scaling
        )
    }
    
    return models


# ==================== MODEL TRAINING & EVALUATION ====================

def train_and_evaluate(model, X_train, X_test, y_train, y_test, needs_scaling=False):
    """
    Train model and evaluate performance.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Scale features if needed
    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'training_time': training_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Cross-validation score (on training set)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, scoring='f1'
    )
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    return metrics, model


# ==================== RESULTS COMPARISON ====================

def compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare all models.
    
    Returns:
        DataFrame with comparison results
    """
    models = get_models()
    results = []
    
    print("\n" + "=" * 70)
    print("🔬 Training & Evaluating Models")
    print("=" * 70)
    
    for model_name, (model, needs_scaling) in models.items():
        print(f"\n⏳ Training {model_name}...")
        
        metrics, trained_model = train_and_evaluate(
            model, X_train, X_test, y_train, y_test, needs_scaling
        )
        
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'CV F1 (mean)': metrics['cv_f1_mean'],
            'CV F1 (std)': metrics['cv_f1_std'],
            'Training Time (s)': metrics['training_time']
        })
        
        print(f"   ✅ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ✅ Precision: {metrics['precision']:.4f}")
        print(f"   ✅ Recall:    {metrics['recall']:.4f}")
        print(f"   ✅ F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ✅ ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   ⏱️  Time:      {metrics['training_time']:.2f}s")
    
    df_results = pd.DataFrame(results)
    return df_results


# ==================== DISPLAY RESULTS ====================

def display_comparison(df_results):
    """Display formatted comparison table."""
    print("\n" + "=" * 70)
    print("📊 FORMAL MODEL COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(df_results.to_string(index=False))
    print()
    
    # Identify best model for each metric
    print("=" * 70)
    print("🏆 BEST MODELS BY METRIC")
    print("=" * 70)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for metric in metrics:
        best_idx = df_results[metric].idxmax()
        best_model = df_results.loc[best_idx, 'Model']
        best_value = df_results.loc[best_idx, metric]
        print(f"  {metric:12s}: {best_model:25s} ({best_value:.4f})")
    
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATION")
    print("=" * 70)
    
    # For VC context, prioritize Recall (don't miss winners)
    best_recall_idx = df_results['Recall'].idxmax()
    recommended_model = df_results.loc[best_recall_idx, 'Model']
    
    print(f"""
  In the VC context, RECALL is most critical (don't miss unicorns).
  
  Recommended Model: {recommended_model}
  
  Rationale:
  - High recall = captures most successful startups
  - Venture capital is asymmetric: 1 unicorn >> 10 failures
  - Missing a winner (false negative) costs more than 
    investing in a failure (false positive)
    """)


# ==================== SAVE RESULTS ====================

def save_results(df_results):
    """Save comparison results to CSV."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "model_comparison.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n💾 Results saved to: {output_path}")


# ==================== MAIN ====================

def main():
    """Run formal model comparison."""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare ML data
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(df)
    
    # Compare models
    df_results = compare_models(X_train, X_test, y_train, y_test)
    
    # Display results
    display_comparison(df_results)
    
    # Save results
    save_results(df_results)
    
    print("\n" + "=" * 70)
    print("✅ Model comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()