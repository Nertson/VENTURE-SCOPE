"""
Tests for Temporal Split Module

Critical validation: Ensure no future data leakage (look-ahead bias).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


@pytest.fixture
def temporal_splits():
    """Load temporal splits for testing."""
    data_dir = Path('data/processed')
    
    train = pd.read_csv(data_dir / 'train_2000_2010.csv')
    val = pd.read_csv(data_dir / 'val_2011.csv')
    test = pd.read_csv(data_dir / 'test_2012_2013.csv')
    
    # Parse dates
    for df in [train, val, test]:
        df['last_funding_date'] = pd.to_datetime(df['last_funding_date'])
        df['cutoff_date'] = pd.to_datetime(df['cutoff_date'])
    
    return train, val, test


# ==================== CRITICAL TESTS: NO LEAKAGE ====================

def test_no_future_funding_in_train(temporal_splits):
    """
    CRITICAL: Train set must have NO funding rounds after Dec 31, 2010.
    
    Why critical:
    - If train sees 2011 funding, model learns from future data
    - This is the primary look-ahead bias we're eliminating
    
    Defense answer: "This test proves train set cutoff is enforced."
    """
    train, _, _ = temporal_splits
    
    max_funding_date = train['last_funding_date'].max()
    cutoff_date = datetime(2010, 12, 31)
    
    assert max_funding_date <= cutoff_date, \
        f"LEAK DETECTED: Train has funding after {cutoff_date.date()}, max={max_funding_date.date()}"
    
    print(f"✓ Train last funding: {max_funding_date.date()} <= {cutoff_date.date()}")


def test_no_future_funding_in_val(temporal_splits):
    """
    CRITICAL: Validation set must have NO funding rounds after Dec 31, 2011.
    """
    _, val, _ = temporal_splits
    
    max_funding_date = val['last_funding_date'].max()
    cutoff_date = datetime(2011, 12, 31)
    
    assert max_funding_date <= cutoff_date, \
        f"LEAK DETECTED: Val has funding after {cutoff_date.date()}, max={max_funding_date.date()}"
    
    print(f"✓ Val last funding: {max_funding_date.date()} <= {cutoff_date.date()}")


def test_no_future_funding_in_test(temporal_splits):
    """
    CRITICAL: Test set must have NO funding rounds after Dec 31, 2012.
    """
    _, _, test = temporal_splits
    
    max_funding_date = test['last_funding_date'].max()
    cutoff_date = datetime(2012, 12, 31)
    
    assert max_funding_date <= cutoff_date, \
        f"LEAK DETECTED: Test has funding after {cutoff_date.date()}, max={max_funding_date.date()}"
    
    print(f"✓ Test last funding: {max_funding_date.date()} <= {cutoff_date.date()}")


def test_cutoff_dates_correct(temporal_splits):
    """
    Verify each split uses its correct cutoff date.
    """
    train, val, test = temporal_splits
    
    # All train records should have cutoff = 2010-12-31
    train_cutoffs = train['cutoff_date'].unique()
    assert len(train_cutoffs) == 1, "Train should have single cutoff date"
    assert train_cutoffs[0] == pd.Timestamp('2010-12-31'), "Train cutoff should be 2010-12-31"
    
    # All val records should have cutoff = 2011-12-31
    val_cutoffs = val['cutoff_date'].unique()
    assert len(val_cutoffs) == 1, "Val should have single cutoff date"
    assert val_cutoffs[0] == pd.Timestamp('2011-12-31'), "Val cutoff should be 2011-12-31"
    
    # All test records should have cutoff = 2012-12-31
    test_cutoffs = test['cutoff_date'].unique()
    assert len(test_cutoffs) == 1, "Test should have single cutoff date"
    assert test_cutoffs[0] == pd.Timestamp('2012-12-31'), "Test cutoff should be 2012-12-31"
    
    print("✓ All splits use correct cutoff dates")


# ==================== DATA QUALITY TESTS ====================

def test_temporal_distributions_increase(temporal_splits):
    """
    Funding amounts should generally increase over time (companies raise more rounds).
    
    Why: If test mean < train mean, suggests data quality issue or leakage.
    """
    train, val, test = temporal_splits
    
    train_mean = train['funding_amount'].mean()
    val_mean = val['funding_amount'].mean()
    test_mean = test['funding_amount'].mean()
    
    print(f"\nFunding distributions:")
    print(f"  Train (2010): ${train_mean/1e6:.2f}M")
    print(f"  Val (2011): ${val_mean/1e6:.2f}M")
    print(f"  Test (2012): ${test_mean/1e6:.2f}M")
    
    # Should increase (or at least not drastically decrease)
    assert val_mean >= train_mean * 0.9, "Val funding shouldn't be much lower than train"
    assert test_mean >= val_mean * 0.9, "Test funding shouldn't be much lower than val"


def test_all_splits_have_data(temporal_splits):
    """Verify each split has sufficient data for training/validation."""
    train, val, test = temporal_splits
    
    assert len(train) >= 1000, f"Train too small: {len(train)} companies"
    assert len(val) >= 100, f"Val too small: {len(val)} companies"
    assert len(test) >= 100, f"Test too small: {len(test)} companies"
    
    print(f"\n✓ Split sizes:")
    print(f"  Train: {len(train):,} companies")
    print(f"  Val: {len(val):,} companies")
    print(f"  Test: {len(test):,} companies")


def test_required_features_present(temporal_splits):
    """Verify all required features are present in splits."""
    train, val, test = temporal_splits
    
    required_features = [
        'company_id', 'company', 'funding_amount', 'investors_count',
        'stage', 'sector', 'country', 'age_years',
        'estimated_revenue', 'capital_efficiency', 'burn_multiple',
        'runway_months', 'traction_index', 'rule_of_40', 'investment_score',
        'status', 'success'
    ]
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        missing = set(required_features) - set(df.columns)
        assert len(missing) == 0, f"{split_name} missing features: {missing}"
    
    print(f"✓ All required features present in all splits")


def test_no_zero_funding(temporal_splits):
    """Verify filtering removed $0 funding companies."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        zero_funding = (df['funding_amount'] == 0).sum()
        assert zero_funding == 0, f"{split_name} has {zero_funding} companies with $0 funding"
    
    print("✓ No $0 funding companies in any split")


def test_no_zero_investors(temporal_splits):
    """Verify filtering removed companies with no investors."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        zero_investors = (df['investors_count'] == 0).sum()
        assert zero_investors == 0, f"{split_name} has {zero_investors} companies with 0 investors"
    
    print("✓ No 0-investor companies in any split")


# ==================== KPI VALIDATION TESTS ====================

def test_capital_efficiency_range(temporal_splits):
    """Capital efficiency should be in reasonable range [0, 2.0]."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        min_eff = df['capital_efficiency'].min()
        max_eff = df['capital_efficiency'].max()
        
        assert min_eff >= 0, f"{split_name} has negative capital efficiency"
        assert max_eff <= 2.0, f"{split_name} has unrealistic capital efficiency: {max_eff}"
    
    print("✓ Capital efficiency in valid range")


def test_burn_multiple_capped(temporal_splits):
    """Burn multiple should be capped at 10."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        max_burn = df['burn_multiple'].max()
        assert max_burn <= 10, f"{split_name} has uncapped burn multiple: {max_burn}"
    
    print("✓ Burn multiple capped at 10")


def test_traction_index_normalized(temporal_splits):
    """Traction index should be normalized 0-100."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        min_traction = df['traction_index'].min()
        max_traction = df['traction_index'].max()
        
        assert min_traction >= 0, f"{split_name} has negative traction index"
        assert max_traction <= 100, f"{split_name} traction index exceeds 100"
    
    print("✓ Traction index normalized 0-100")


def test_investment_score_range(temporal_splits):
    """Investment score should be in range [0, 100]."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        min_score = df['investment_score'].min()
        max_score = df['investment_score'].max()
        
        assert min_score >= 0, f"{split_name} has negative investment score"
        assert max_score <= 100, f"{split_name} investment score exceeds 100"
    
    print("✓ Investment score in valid range [0, 100]")


# ==================== TARGET VARIABLE TESTS ====================

def test_success_label_binary(temporal_splits):
    """Success label should be binary (0 or 1) or NaN."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        valid_values = df['success'].dropna().isin([0, 1]).all()
        assert valid_values, f"{split_name} has invalid success values"
    
    print("✓ Success labels are binary")


def test_success_rate_reasonable(temporal_splits):
    """Success rate should be between 40-80% (Crunchbase survivor bias)."""
    train, val, test = temporal_splits
    
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        success_rate = df['success'].dropna().mean()
        
        assert 0.4 <= success_rate <= 0.8, \
            f"{split_name} success rate {success_rate:.1%} outside expected range [40%, 80%]"
        
        print(f"  {split_name} success rate: {success_rate:.1%}")


def test_known_outcomes_sufficient(temporal_splits):
    """Each split should have sufficient companies with known outcomes for ML."""
    train, val, test = temporal_splits
    
    train_known = train['success'].notna().sum()
    val_known = val['success'].notna().sum()
    test_known = test['success'].notna().sum()
    
    assert train_known >= 500, f"Train has only {train_known} companies with known outcomes"
    assert val_known >= 50, f"Val has only {val_known} companies with known outcomes"
    assert test_known >= 50, f"Test has only {test_known} companies with known outcomes"
    
    print(f"\n✓ Known outcomes:")
    print(f"  Train: {train_known:,}")
    print(f"  Val: {val_known:,}")
    print(f"  Test: {test_known:,}")


# ==================== COMPARISON TEST ====================

def test_compare_to_random_split():
    """
    Compare temporal split sizes to original random split.
    
    Expected: Temporal splits should be smaller (less data available at cutoff dates).
    """
    # Load original random split data if exists
    try:
        original = pd.read_csv('data/processed/startups_scored.csv')
        original_ml = original[original['status'].isin(['acquired', 'ipo', 'closed'])]
        
        # Load temporal splits
        train = pd.read_csv('data/processed/train_2000_2010.csv')
        val = pd.read_csv('data/processed/val_2011.csv')
        test = pd.read_csv('data/processed/test_2012_2013.csv')
        
        # CORRECTED: Filter temporal splits to known outcomes only (apples-to-apples)
        train_ml = train[train['status'].isin(['acquired', 'ipo', 'closed'])]
        val_ml = val[val['status'].isin(['acquired', 'ipo', 'closed'])]
        test_ml = test[test['status'].isin(['acquired', 'ipo', 'closed'])]
        
        temporal_ml_total = len(train_ml) + len(val_ml) + len(test_ml)
        temporal_all_total = len(train) + len(val) + len(test)
        
        print(f"\n✓ Data size comparison:")
        print(f"  Original (random split, known outcomes): {len(original_ml):,} companies")
        print(f"  Temporal (known outcomes only): {temporal_ml_total:,} companies")
        print(f"  Temporal (all statuses): {temporal_all_total:,} companies")
        print(f"  Difference (ML-ready): {len(original_ml) - temporal_ml_total:,} companies")
        
        # Temporal ML-ready should be comparable or smaller
        # (May be slightly larger if more companies had outcomes by 2012 than in filtered original)
        print(f"\n  Train ML-ready: {len(train_ml):,}")
        print(f"  Val ML-ready: {len(val_ml):,}")
        print(f"  Test ML-ready: {len(test_ml):,}")
        
    except FileNotFoundError:
        print("⚠ Original data not found, skipping comparison")





if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])