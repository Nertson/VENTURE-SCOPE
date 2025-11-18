"""
Enriched Crunchbase loader for Venture-Scope.

CRUNCHBASE-SPECIFIC LOADER - Requires multiple CSV files.
For generic CSV loading, use loaders.py instead.

This module loads objects.csv and enriches it with:
- Funding stages from funding_rounds.csv
- Investor counts from investments.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Optional

# Map Crunchbase funding types to standard stages
STAGE_MAP = {
    "seed": "Seed",
    "angel": "Angel",
    "series-a": "Series A",
    "series-a+": "Series A",
    "series-b": "Series B",
    "series-b+": "Series B",
    "series-c": "Series C",
    "series-c+": "Series C",
    "series-d": "Series D+",
    "series-e": "Series D+",
    "series-f": "Series D+",
    "series-g": "Series D+",
    "venture": "Series A",
    "private-equity": "Series D+",
}


def load_enriched_startups(
    data_dir: str | Path,
    filter_funded: bool = True,
    min_funding: float = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load Crunchbase data with enrichment from multiple sources.
    
    Args:
        data_dir: Directory containing Crunchbase CSV files
        filter_funded: Keep only companies with funding > min_funding
        min_funding: Minimum funding threshold
        verbose: Print progress
    
    Returns:
        Enriched DataFrame with stage and investors_count
    
    Example:
        >>> df = load_enriched_startups("data/raw/")
        >>> df[['company', 'stage', 'investors_count']].head()
    """
    data_dir = Path(data_dir)
    
    if verbose:
        print("📂 Loading enriched Crunchbase data...\n")
    
    # ==================== LOAD OBJECTS (COMPANIES) ====================
    
    if verbose:
        print("  ├─ Step 1/4: Loading objects.csv (companies)...")
    
    objects = pd.read_csv(
        data_dir / "objects.csv",
        low_memory=False
    )
    
    # Filter only companies
    if 'entity_type' in objects.columns:
        original = len(objects)
        objects = objects[objects['entity_type'] == 'Company'].copy()
        if verbose:
            print(f"    ✓ Filtered to {len(objects):,} companies (removed {original - len(objects):,} non-companies)")
    
    # Select relevant columns
    companies = objects[[
        'id', 'name', 'category_code', 'country_code', 
        'founded_at', 'funding_total_usd', 'status'
    ]].copy()
    
    # ==================== LOAD FUNDING ROUNDS ====================
    
    if verbose:
        print("  ├─ Step 2/4: Loading funding_rounds.csv (stages)...")
    
    rounds = pd.read_csv(
        data_dir / "funding_rounds.csv",
        low_memory=False
    )
    
    if verbose:
        print(f"    ✓ Loaded {len(rounds):,} funding rounds")
    
    # Get the LAST (most recent) round for each company
    rounds_sorted = rounds.sort_values('funded_at')
    last_rounds = rounds_sorted.groupby('object_id').last().reset_index()
    last_rounds = last_rounds[['object_id', 'funding_round_type', 'funded_at']]
    last_rounds.columns = ['id', 'last_round_type', 'last_funding_date']
    
    # Standardize stage names
    last_rounds['stage'] = last_rounds['last_round_type'].str.lower().str.strip()
    last_rounds['stage'] = last_rounds['stage'].map(STAGE_MAP)
    
    if verbose:
        print(f"    ✓ Extracted stages for {len(last_rounds):,} companies")
    
    # ==================== LOAD INVESTMENTS (INVESTOR COUNT) ====================
    
    if verbose:
        print("  ├─ Step 3/4: Loading investments.csv (investor counts)...")
    
    investments = pd.read_csv(
        data_dir / "investments.csv",
        low_memory=False
    )
    
    if verbose:
        print(f"    ✓ Loaded {len(investments):,} investment records")
    
    # Count unique investors per company
    investor_counts = investments.groupby('funded_object_id')['investor_object_id'].nunique().reset_index()
    investor_counts.columns = ['id', 'investors_count']
    
    if verbose:
        print(f"    ✓ Counted investors for {len(investor_counts):,} companies")
    
    # ==================== MERGE ALL DATA ====================
    
    if verbose:
        print("  └─ Step 4/4: Merging all data sources...")
    
    # Merge companies with stages
    df = companies.merge(last_rounds[['id', 'stage', 'last_funding_date']], on='id', how='left')
    
    # Merge with investor counts
    df = df.merge(investor_counts, on='id', how='left')
    
    if verbose:
        print(f"    ✓ Merged data: {len(df):,} companies")
    
    # ==================== STANDARDIZE & CLEAN ====================
    
    # Extract founded year
    df['founded_year'] = pd.to_datetime(df['founded_at'], errors='coerce').dt.year
    
    # Rename columns to standard names
    df = df.rename(columns={
        'name': 'company',
        'category_code': 'sector',
        'country_code': 'country',
        'funding_total_usd': 'funding_amount'
    })
    
    # Select final columns
    result = df[[
        'company', 'stage', 'country', 'sector', 
        'funding_amount', 'investors_count', 'founded_year',
        'status', 'last_funding_date'
    ]].copy()
    
    # Convert types
    result['company'] = result['company'].astype('string')
    result['stage'] = result['stage'].astype('string')
    result['country'] = result['country'].astype('string')
    result['sector'] = result['sector'].astype('string')
    result['status'] = result['status'].astype('string')
    
   # ==================== FILTER ====================

    if filter_funded:
        before = len(result)
        
        # CRITICAL: Always filter > 0 to exclude missing data ($0)
        # Most $0 values in Crunchbase represent missing data, not true $0 funding
        result = result[result['funding_amount'] > 0].copy()
        
        # If min_funding is specified and > 0, apply additional threshold
        if min_funding > 0:
            result = result[result['funding_amount'] >= min_funding].copy()
        
        if verbose:
            threshold_text = f"${min_funding:,.0f}" if min_funding > 0 else "$0"
            print(f"\n🔍 Filtered: Kept {len(result):,} companies with funding > {threshold_text}")
            print(f"   Removed: {before - len(result):,} companies")
    
    # ==================== REPORT ====================
    
    if verbose:
        print(f"\n✨ Final dataset: {len(result):,} companies × {len(result.columns)} columns")
        _data_quality_report(result)
    
    return result


def _data_quality_report(df: pd.DataFrame) -> None:
    """Print data quality summary."""
    print("\n📊 Data Quality Report:")
    print("=" * 60)
    
    for col in ['company', 'stage', 'country', 'sector', 'funding_amount', 'investors_count', 'founded_year']:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = (missing / len(df)) * 100
            
            if pct == 0:
                icon = "✅"
            elif pct < 20:
                icon = "🟢"
            elif pct < 50:
                icon = "🟡"
            else:
                icon = "🔴"
            
            print(f"  {icon} {col:25s}: {missing:7,} missing ({pct:5.1f}%)")
    
    print("=" * 60)


# ==================== TESTING ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/raw/"
    
    print(f"Loading enriched data from: {data_dir}\n")
    
    df = load_enriched_startups(
        data_dir,
        filter_funded=True,
        min_funding=0  # Keep all with funding > 0
    )
    
    print(f"\n✅ Successfully loaded {len(df):,} companies!")
    print(f"\n📋 Sample (first 10 rows):\n")
    print(df[['company', 'stage', 'country', 'sector', 'funding_amount', 'investors_count']].head(10).to_string())
    
    # Stage distribution
    print(f"\n📊 Stage Distribution:\n")
    print(df['stage'].value_counts().head(10))
    
    # Save to processed
    output_path = Path("data/processed/startups_enriched.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")