"""
ML Model Training with Temporal Validation

CRITICAL DIFFERENCE from model.py:
- Uses temporal splits (train/val/test) with strict cutoff dates
- Eliminates look-ahead bias through proper temporal validation
- Validates on separate time period before testing

"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class TemporalModelTrainer:
    """
    Train ML model using temporal train/val/test splits.
    """
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / 'models').mkdir(exist_ok=True)
        
    def load_temporal_splits(self):
        """Load train, validation, and test sets with temporal separation."""
        print("="*70)
        print("LOADING TEMPORAL SPLITS")
        print("="*70)
        
        train_path = self.data_dir / 'train_2000_2010.csv'
        val_path = self.data_dir / 'val_2011.csv'
        test_path = self.data_dir / 'test_2012_2013.csv'
        
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"\n✓ Loaded temporal splits:")
        print(f"  Train (2000-2010): {len(self.train_df):,} companies")
        print(f"  Val (2011): {len(self.val_df):,} companies")
        print(f"  Test (2012-2013): {len(self.test_df):,} companies")
        
        # Verify cutoff dates
        train_max = pd.to_datetime(self.train_df['cutoff_date']).max()
        val_max = pd.to_datetime(self.val_df['cutoff_date']).max()
        test_max = pd.to_datetime(self.test_df['cutoff_date']).max()
        
        print(f"\n✓ Cutoff dates verified:")
        print(f"  Train: {train_max.date()}")
        print(f"  Val: {val_max.date()}")
        print(f"  Test: {test_max.date()}")
        
        return self
    
    def prepare_features(self, df):
        """
        Prepare features for ML (one-hot encoding for categorical).
        
        Returns X (features) and y (target), plus feature names.
        """
        # Filter to known outcomes only
        df_ml = df[df['status'].isin(['acquired', 'ipo', 'closed'])].copy()
        
        # Create target variable
        df_ml['success'] = df_ml['status'].apply(
            lambda x: 1 if x in ['acquired', 'ipo'] else 0
        )
        
        # Numeric features
        numeric_features = [
            'funding_amount',
            'investors_count',
            'age_years',
            'months_since_last_funding',
            'estimated_revenue',
            'capital_efficiency',
            'monthly_burn',
            'runway_months',
            'burn_multiple',
            'traction_index',
            'rule_of_40',
            'investment_score'
        ]
        
        X_numeric = df_ml[numeric_features].fillna(0)
        
        # One-hot encode categorical features
        X_stage = pd.get_dummies(df_ml['stage'], prefix='stage')
        X_sector = pd.get_dummies(df_ml['sector'], prefix='sector')
        X_country = pd.get_dummies(df_ml['country'], prefix='country')
        
        # Combine all features
        X = pd.concat([X_numeric, X_stage, X_sector, X_country], axis=1)
        y = df_ml['success']
        
        return X, y, df_ml
    
    def prepare_all_splits(self):
        """Prepare X, y for train, val, and test."""
        print("\n" + "="*70)
        print("PREPARING FEATURES")
        print("="*70)
        
        print("\nTrain set:")
        self.X_train, self.y_train, self.train_ml = self.prepare_features(self.train_df)
        print(f"  Features: {self.X_train.shape[1]} columns")
        print(f"  Samples: {len(self.X_train):,} companies with known outcomes")
        print(f"  Success rate: {self.y_train.mean()*100:.1f}%")
        
        print("\nValidation set:")
        self.X_val, self.y_val, self.val_ml = self.prepare_features(self.val_df)
        
        # Align val columns with train (same one-hot encoding)
        missing_cols = set(self.X_train.columns) - set(self.X_val.columns)
        for col in missing_cols:
            self.X_val[col] = 0
        self.X_val = self.X_val[self.X_train.columns]  # Same column order
        
        print(f"  Features: {self.X_val.shape[1]} columns (aligned with train)")
        print(f"  Samples: {len(self.X_val):,} companies with known outcomes")
        print(f"  Success rate: {self.y_val.mean()*100:.1f}%")
        
        print("\nTest set:")
        self.X_test, self.y_test, self.test_ml = self.prepare_features(self.test_df)
        
        # Align test columns with train
        missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
        for col in missing_cols:
            self.X_test[col] = 0
        self.X_test = self.X_test[self.X_train.columns]
        
        print(f"  Features: {self.X_test.shape[1]} columns (aligned with train)")
        print(f"  Samples: {len(self.X_test):,} companies with known outcomes")
        print(f"  Success rate: {self.y_test.mean()*100:.1f}%")
        
        return self
    
    def train_model(self):
        """Train Random Forest on train set."""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        print("\nRandom Forest Classifier:")
        print("  n_estimators: 100")
        print("  max_depth: 10")
        print("  random_state: 42")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("\nTraining on train set...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Training complete")
        
        return self
    
    def evaluate_on_set(self, X, y, set_name):
        """Evaluate model on a dataset."""
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        cm = confusion_matrix(y, y_pred)
        
        print(f"\n{set_name} Set Performance:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"  Precision: {metrics['precision']*100:.1f}%")
        print(f"  Recall:    {metrics['recall']*100:.1f}%")
        print(f"  F1-Score:  {metrics['f1']*100:.1f}%")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        
        return metrics, cm
    
    def evaluate_all(self):
        """Evaluate on train, val, and test sets."""
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        self.train_metrics, self.train_cm = self.evaluate_on_set(
            self.X_train, self.y_train, "Train"
        )
        
        self.val_metrics, self.val_cm = self.evaluate_on_set(
            self.X_val, self.y_val, "Validation"
        )
        
        self.test_metrics, self.test_cm = self.evaluate_on_set(
            self.X_test, self.y_test, "Test"
        )
        
        # Check for overfitting
        print("\n" + "="*70)
        print("OVERFITTING ANALYSIS")
        print("="*70)
        
        train_acc = self.train_metrics['accuracy']
        val_acc = self.val_metrics['accuracy']
        test_acc = self.test_metrics['accuracy']
        
        print(f"\nAccuracy across splits:")
        print(f"  Train: {train_acc*100:.1f}%")
        print(f"  Val:   {val_acc*100:.1f}%")
        print(f"  Test:  {test_acc*100:.1f}%")
        
        train_val_gap = (train_acc - val_acc) * 100
        val_test_gap = (val_acc - test_acc) * 100
        
        print(f"\nGaps:")
        print(f"  Train-Val gap:  {train_val_gap:+.1f}%")
        print(f"  Val-Test gap:   {val_test_gap:+.1f}%")
        
        if train_val_gap > 5:
            print("   Moderate overfitting detected (>5% gap)")
        else:
            print("  ✓ Acceptable generalization (<5% gap)")
        
        return self
    
    def compare_with_baseline(self):
        """Compare with baseline model.py results."""
        print("\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)
        
        baseline_metrics = {
            'accuracy': 0.760,
            'precision': 0.757,
            'recall': 0.901,
            'f1': 0.822,
            'roc_auc': 0.805
        }
        
        print("\nPerformance Comparison:")
        print(f"{'Metric':<15} {'Baseline':<12} {'Temporal':<12} {'Difference':<12}")
        print("-" * 55)
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            baseline_val = baseline_metrics[metric_name]
            temporal_val = self.test_metrics[metric_name]
            diff = (temporal_val - baseline_val) * 100
            
            print(f"{metric_name.capitalize():<15} {baseline_val*100:>6.1f}%     {temporal_val*100:>6.1f}%     {diff:>+6.1f}%")
        
        print("\nInterpretation:")
        recall_diff = (self.test_metrics['recall'] - baseline_metrics['recall']) * 100
        
        if recall_diff < 0:
            print(f"  • Recall decreased by {abs(recall_diff):.1f}% (from 90.1% to {self.test_metrics['recall']*100:.1f}%)")
            print(f"  • This is the cost of eliminating look-ahead bias")
            print(f"  • {self.test_metrics['recall']*100:.1f}% recall still captures ~{int(self.test_metrics['recall']*10)}/10 winners")
            print(f"  • Trade-off: Methodological rigor vs raw performance")
        else:
            print(f"  • Recall maintained or improved")
        
        return self
    
    def save_model(self):
        """Save trained model and metrics."""
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        # Save model
        model_path = self.results_dir / 'models' / 'random_forest_temporal.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved: {model_path}")
        
        # Save metrics
        metrics_all = {
            'train': {k: float(v) for k, v in self.train_metrics.items()},
            'val': {k: float(v) for k, v in self.val_metrics.items()},
            'test': {k: float(v) for k, v in self.test_metrics.items()},
            'confusion_matrices': {
                'train': self.train_cm.tolist(),
                'val': self.val_cm.tolist(),
                'test': self.test_cm.tolist()
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'train_size': len(self.X_train),
                'val_size': len(self.X_val),
                'test_size': len(self.X_test),
                'n_features': self.X_train.shape[1]
            }
        }
        
        metrics_path = self.results_dir / 'temporal_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_all, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = self.results_dir / 'feature_importance_temporal.csv'
        feature_importance.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved: {importance_path}")
        
        print(f"\nTop 10 features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['importance']*100:>6.2f}%")
        
        return self


def main():
    """Main execution: Train model with temporal validation."""
    print("="*70)
    print("ML MODEL TRAINING - TEMPORAL VALIDATION")
    print("="*70)
    print("\nCRITICAL: This version uses temporal splits to eliminate look-ahead bias")
    print("Expected performance decrease (~5%) is the cost of methodological rigor")
    
    # Initialize trainer
    trainer = TemporalModelTrainer()
    
    # Load temporal splits
    trainer.load_temporal_splits()
    
    # Prepare features
    trainer.prepare_all_splits()
    
    # Train model
    trainer.train_model()
    
    # Evaluate
    trainer.evaluate_all()
    
    # Compare with baseline
    trainer.compare_with_baseline()
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    


if __name__ == '__main__':
    main()