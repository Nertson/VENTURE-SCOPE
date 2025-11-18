"""
Data loading and standardization module for Venture-Scope.

GENERIC LOADER - Works with any CSV file with startup data.
For Crunchbase-specific enriched loading, use loaders_enriched.py instead.

This module provides robust CSV loading with automatic schema harmonization,
handling various naming conventions and ensuring data quality.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

# ==================== CONSTANTS ====================

CANONICAL_COLS = (
    "company",
    "stage",
    "country",
    "sector",
    "funding_amount",
    "investors_count",
    "founded_year",
)

STAGE_MAP = {
    "pre-seed": "Pre-Seed",
    "seed": "Seed",
    "angel": "Angel",
    "series a": "Series A",
    "series b": "Series B",
    "series c": "Series C",
    "series d": "Series D+",
    "series e": "Series D+",
    "series f": "Series D+",
}

# ==================== HELPER FUNCTIONS ====================

def _standardize_stage(x: Optional[str]) -> Optional[str]:
    """
    Standardize funding stage nomenclature.
    
    Args:
        x: Raw stage value
    
    Returns:
        Standardized stage name or None
    
    Examples:
        >>> _standardize_stage("seed")
        'Seed'
        >>> _standardize_stage("SERIES A")
        'Series A'
        >>> _standardize_stage("")
        None
    """
    if not isinstance(x, str) or not x.strip():
        return None
    return STAGE_MAP.get(x.lower().strip(), x.strip().title())


def _coalesce(df: pd.DataFrame, cands: list[str], new_col: str, verbose: bool = True) -> pd.Series:
    """
    Coalesce multiple candidate columns into one.
    
    Takes the first non-null value from candidate columns in order.
    
    Args:
        df: Source DataFrame
        cands: List of candidate column names (in priority order)
        new_col: Name for the resulting column
        verbose: Whether to print which columns were found
    
    Returns:
        Series with coalesced values
    
    Example:
        >>> s = _coalesce(df, ["sector", "category", "industry"], "sector")
    """
    s = pd.Series([None] * len(df))
    found_cols = []
    
    for c in cands:
        if c in df.columns:
            s = s.fillna(df[c])
            found_cols.append(c)
    
    if verbose:
        if found_cols:
            print(f"  ℹ️  Coalesced '{new_col}' from: {', '.join(found_cols)}")
        else:
            print(f"  ⚠️  No candidate columns found for '{new_col}'")
    
    return s.rename(new_col)


def _data_quality_report(df: pd.DataFrame) -> None:
    """
    Print data quality summary.
    
    Args:
        df: DataFrame to analyze
    """
    print("\n📊 Data Quality Report:")
    print("=" * 60)
    
    for col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        
        if pct > 0:
            print(f"  {col:20s}: {missing:6,} missing ({pct:5.1f}%)")
        else:
            print(f"  {col:20s}: ✅ Complete")
    
    print("=" * 60)


# ==================== MAIN LOADER ====================

def load_startups_csv(path: str | Path, verbose: bool = True, filter_funded: bool = False) -> pd.DataFrame:
    """
    Load and standardize a startup dataset from CSV.
    
    Performs comprehensive data cleaning:
    - Identifies and renames company column variants
    - Standardizes funding stage nomenclature
    - Coalesces sector/country from multiple column names
    - Converts numeric fields with error handling
    - Ensures all canonical columns exist
    - Filters non-company entities (people, investors, products)
    - Optionally filters companies with funding_amount > $0
    
    Data Quality Decisions:
    ----------------------
    1. Entity Filtering: Only keeps entity_type == 'Company'
       - Removes investors, people, financial orgs, products
       - Rationale: Focus on actual startup companies
    
    2. Funding Filtering (if filter_funded=True):
       - Removes companies with funding_amount <= $0
       - Rationale: Most $0 values represent missing data in Crunchbase,
         not true bootstrapped companies. Our VC-focused KPIs require
         funding data.
       - Limitation: Excludes successful bootstrapped companies
    
    Args:
        path: Path to the CSV file
        verbose: Whether to print progress messages
        filter_funded: If True, keep only companies with funding > $0
    
    Returns:
        DataFrame with standardized columns: company, stage, country,
        sector, funding_amount, investors_count, founded_year
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is empty
    
    Example:
        >>> # Load all companies
        >>> df_all = load_startups_csv("data/raw/objects.csv")
        >>> 
        >>> # Load only funded companies
        >>> df_funded = load_startups_csv("data/raw/objects.csv", filter_funded=True)
    """
    # Validate file existence
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    if verbose:
        print(f"📂 Loading startups from: {path.name}")
    
    # Load CSV with low_memory=False to avoid dtype warnings
    df = pd.read_csv(path, low_memory=False)
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")
    
    if verbose:
        print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

     # CRITICAL: Filter only companies (not people, investors, products, etc.)
    if 'entity_type' in df.columns:
        original_count = len(df)
        df = df[df['entity_type'] == 'Company'].copy()
        if verbose:
            filtered_count = original_count - len(df)
            print(f"  🔍 Filtered out {filtered_count:,} non-company entities (kept only Companies)")
    
    # Find and rename company column
    if "company" not in df.columns:
        for cand in ("company", "organization", "startup", "name"):
            if cand in df.columns:
                df = df.rename(columns={cand: "company"})
                if verbose:
                    print(f"  ℹ️  Renamed '{cand}' → 'company'")
                break
    
    # Standardize stage
    if "stage" not in df.columns:
        df["stage"] = None
    df["stage"] = df["stage"].apply(_standardize_stage)
    
    # Harmonize sector and country
    if "sector" not in df.columns:
        df["sector"] = _coalesce(df, ["sector", "category_code", "category", "industry"], "sector", verbose)
    if "country" not in df.columns:
        df["country"] = _coalesce(df, ["country", "country_code", "hq_country"], "country", verbose)
    
    # Convert numeric columns with smart mapping
    numeric_mapping = {
        "funding_amount": ["funding_amount", "funding_total_usd", "raised_amount_usd", "funding_total"],
        "investors_count": ["investors_count", "investor_count", "participants"],
        "founded_year": ["founded_year", "founded_at"]
    }
    
    for target_col, candidates in numeric_mapping.items():
        if target_col not in df.columns:
            for cand in candidates:
                if cand in df.columns:
                    if cand == "founded_at":
                        # Extract year from date
                        df[target_col] = pd.to_datetime(df[cand], errors="coerce").dt.year
                        if verbose:
                            print(f"  ℹ️  Extracted year from '{cand}' → '{target_col}'")
                    else:
                        df[target_col] = pd.to_numeric(df[cand], errors="coerce")
                        if verbose:
                            print(f"  ℹ️  Mapped '{cand}' → '{target_col}'")
                    break
        else:
            # Column already exists, just convert type
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    # Ensure all canonical columns exist
    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    
    # Select and clean
    out = df[list(CANONICAL_COLS)].copy()
    
    for c in ("company", "stage", "country", "sector"):
        out[c] = out[c].astype("string").str.strip()
    
    # Optional: Filter only funded companies
    if filter_funded:
        if verbose:
            before_filter = len(out)
    
    # FILTERING DECISION: Remove companies with funding_amount <= 0
    # 
    # Rationale:
    # - Crunchbase $0 values mostly represent MISSING DATA, not bootstrapped companies
    # - Our KPIs (Rule of 40, Burn Multiple, Capital Efficiency) require funding data
    # - Project scope: VC investment decision-making (not bootstrapped companies)
    # 
    # Trade-offs:
    # - ✅ Cleaner dataset for VC-specific analysis
    # - ✅ Enables calculation of funding-dependent KPIs
    # - ❌ Introduces selection bias (excludes bootstrapped successes)
    # - ❌ Reduces dataset size (~78% of companies filtered out)
    # 
    # This decision is documented in METHODOLOGY.md and the technical report.
    
    out = out[out['funding_amount'] > 0].copy()
    
    if verbose:
        after_filter = len(out)
        removed = before_filter - after_filter
        print(f"\n🔍 Filtered: Kept {after_filter:,} companies with funding > $0")
        print(f"   Removed: {removed:,} companies with $0 or missing funding")
    
    if verbose:
        print(f"✨ Standardized to {len(out):,} rows × {len(out.columns)} columns")
        _data_quality_report(out)
    
    return out


# ==================== TESTING ====================

if __name__ == "__main__":
    # Test rapide si le fichier est exécuté directement
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        
        # Check if --filter flag is present
        filter_funded = "--filter" in sys.argv
        
        df = load_startups_csv(test_path, filter_funded=filter_funded)
        print(f"\n✅ Successfully loaded and standardized {len(df):,} startups!")
        print(f"\n📋 Sample (first 5 rows):\n")
        print(df.head())
        
        # Save to processed directory
        output_path = Path("data/processed/startups_clean.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print("Usage: python loaders.py <path_to_csv> [--filter]")
        print("  --filter: Keep only companies with funding data")