from __future__ import annotations

"""
Venture Capital Scoring Engine for Venture-Scope.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

print("=== SCORING ENGINE LOADED ===")

DEFAULT_WEIGHTS = {
    'rule_of_40': 0.25,
    'traction_index': 0.25,
    'capital_efficiency': 0.20,
    'burn_multiple': 0.15,
    'runway_months': 0.15,
}

def normalize_to_100(series: pd.Series, min_val: float = None, max_val: float = None) -> pd.Series:
    """Normalize series to 0-100 scale."""
    if min_val is None:
        min_val = series.min()
    if max_val is None:
        max_val = series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)
    normalized = (series - min_val) / (max_val - min_val) * 100
    return normalized.clip(0, 100)

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all KPIs to 0-100 scale."""
    print("  Normalizing KPIs...")
    result = df.copy()
    
    # Rule of 40: clip to 0-100
    if 'rule_of_40' in df.columns:
        result['rule_of_40_norm'] = df['rule_of_40'].clip(0, 100)
    
    # Traction Index: already 0-100
    if 'traction_index' in df.columns:
        result['traction_index_norm'] = df['traction_index'].clip(0, 100)
    
    # Capital Efficiency: 0-1 -> 0-100
    if 'capital_efficiency' in df.columns:
        capped = df['capital_efficiency'].clip(0, 1.0)
        result['capital_efficiency_norm'] = capped * 100
    
    # Burn Multiple: LOWER is better, so invert
    if 'burn_multiple' in df.columns:
        inverted = 1 / df['burn_multiple'].clip(lower=0.1)
        result['burn_multiple_norm'] = normalize_to_100(inverted, min_val=0.1, max_val=3.0)
    
    # Runway: more is better
    if 'runway_months' in df.columns:
        result['runway_months_norm'] = normalize_to_100(
            df['runway_months'].clip(0, 24), 
            min_val=0, 
            max_val=24
        )
    
    print("  KPIs normalized!")
    return result

def calculate_investment_score(
    df: pd.DataFrame, 
    weights: Optional[Dict[str, float]] = None, 
    verbose: bool = True
) -> pd.DataFrame:
    """Calculate unified investment score (0-100)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    if verbose:
        print("\nCalculating Investment Scores...")
        print(f"  Weights: Rule40={weights['rule_of_40']:.0%}, Traction={weights['traction_index']:.0%}, CapEff={weights['capital_efficiency']:.0%}, Burn={weights['burn_multiple']:.0%}, Runway={weights['runway_months']:.0%}")
    
    result = normalize_kpis(df)
    
    print("  Computing weighted score...")
    score = pd.Series(0.0, index=result.index)
    
    if 'rule_of_40_norm' in result.columns:
        score += result['rule_of_40_norm'] * weights.get('rule_of_40', 0)
    if 'traction_index_norm' in result.columns:
        score += result['traction_index_norm'] * weights.get('traction_index', 0)
    if 'capital_efficiency_norm' in result.columns:
        score += result['capital_efficiency_norm'] * weights.get('capital_efficiency', 0)
    if 'burn_multiple_norm' in result.columns:
        score += result['burn_multiple_norm'] * weights.get('burn_multiple', 0)
    if 'runway_months_norm' in result.columns:
        score += result['runway_months_norm'] * weights.get('runway_months', 0)
    
    result['investment_score'] = score.round(2)
    
    if verbose:
        print(f"  Investment scores calculated!")
        print(f"  Score range: {score.min():.2f} - {score.max():.2f}")
        print(f"  Mean score: {score.mean():.2f}")
    
    return result

def rank_startups(
    df: pd.DataFrame, 
    score_col: str = 'investment_score', 
    ascending: bool = False
) -> pd.DataFrame:
    """Rank startups by score."""
    print("\nRanking startups...")
    result = df.copy()
    result = result.sort_values(score_col, ascending=ascending)
    result['rank'] = range(1, len(result) + 1)
    print(f"  Ranked {len(result):,} startups")
    return result

def get_top_startups(
    df: pd.DataFrame, 
    n: int = 100, 
    score_col: str = 'investment_score'
) -> pd.DataFrame:
    """Get top N startups."""
    ranked = rank_startups(df, score_col=score_col)
    return ranked.head(n)

def score_breakdown(df: pd.DataFrame, company_name: str) -> None:
    """Print detailed score breakdown for a company."""
    company = df[df['company'] == company_name]
    
    if len(company) == 0:
        print(f"Company not found: {company_name}")
        return
    
    company = company.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"SCORE BREAKDOWN: {company_name}")
    print(f"{'='*70}")
    
    print(f"\nFINAL INVESTMENT SCORE: {company['investment_score']:.2f}/100")
    
    if 'rank' in company:
        print(f"RANK: #{int(company['rank']):,}")
    
    print(f"\nKPI COMPONENTS:")
    print("-" * 70)
    
    # Rule of 40
    if 'rule_of_40' in company:
        raw = company['rule_of_40']
        norm = company.get('rule_of_40_norm', raw)
        contribution = norm * DEFAULT_WEIGHTS['rule_of_40']
        print(f"  Rule of 40:         {raw:>8.2f} -> {norm:>6.2f}/100 (x25% = {contribution:>6.2f})")
    
    # Traction Index
    if 'traction_index' in company:
        raw = company['traction_index']
        norm = company.get('traction_index_norm', raw)
        contribution = norm * DEFAULT_WEIGHTS['traction_index']
        print(f"  Traction Index:     {raw:>8.2f} -> {norm:>6.2f}/100 (x25% = {contribution:>6.2f})")
    
    # Capital Efficiency
    if 'capital_efficiency' in company:
        raw = company['capital_efficiency']
        norm = company.get('capital_efficiency_norm', raw * 100)
        contribution = norm * DEFAULT_WEIGHTS['capital_efficiency']
        print(f"  Capital Efficiency: {raw:>8.2f} -> {norm:>6.2f}/100 (x20% = {contribution:>6.2f})")
    
    # Burn Multiple
    if 'burn_multiple' in company:
        raw = company['burn_multiple']
        norm = company.get('burn_multiple_norm', (1/raw) * 50)
        contribution = norm * DEFAULT_WEIGHTS['burn_multiple']
        print(f"  Burn Multiple:      {raw:>8.2f} -> {norm:>6.2f}/100 (x15% = {contribution:>6.2f})")
    
    # Runway
    if 'runway_months' in company:
        raw = company['runway_months']
        norm = company.get('runway_months_norm', raw / 24 * 100)
        contribution = norm * DEFAULT_WEIGHTS['runway_months']
        print(f"  Runway (months):    {raw:>8.2f} -> {norm:>6.2f}/100 (x15% = {contribution:>6.2f})")
    
    print("-" * 70)
    
    # Basic Info
    print(f"\nBASIC INFO:")
    print(f"  Stage:     {company.get('stage', 'N/A')}")
    print(f"  Country:   {company.get('country', 'N/A')}")
    print(f"  Sector:    {company.get('sector', 'N/A')}")
    print(f"  Funding:   ${company.get('funding_amount', 0):,.0f}")
    print(f"  Investors: {company.get('investors_count', 0):.0f}")
    
    print(f"{'='*70}\n")

def scoring_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for scores."""
    if 'investment_score' not in df.columns:
        print("No investment_score column found")
        return
    
    scores = df['investment_score']
    
    print(f"\n{'='*70}")
    print("INVESTMENT SCORE SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nScore Distribution:")
    print(f"  Mean:      {scores.mean():>8.2f}")
    print(f"  Median:    {scores.quantile(0.50):>8.2f}")
    print(f"  Std Dev:   {scores.std():>8.2f}")
    print(f"  Min:       {scores.min():>8.2f}")
    print(f"  25th:      {scores.quantile(0.25):>8.2f}")
    print(f"  75th:      {scores.quantile(0.75):>8.2f}")
    print(f"  Max:       {scores.max():>8.2f}")
    
    print(f"\nScore Ranges:")
    ranges = [
        (80, 100, "Excellent (Top Tier)"),
        (60, 80, "Strong (High Potential)"),
        (40, 60, "Moderate (Average)"),
        (20, 40, "Weak (Below Average)"),
        (0, 20, "Poor (High Risk)")
    ]
    
    for min_score, max_score, label in ranges:
        count = ((scores >= min_score) & (scores < max_score)).sum()
        pct = count / len(scores) * 100
        print(f"  {label:25s}: {count:>5,} ({pct:>5.1f}%)")
    
    print(f"{'='*70}\n")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n=== MAIN EXECUTION STARTED ===\n")
    
    from pathlib import Path
    
    input_file = Path("data/processed/startups_with_kpis.csv")
    
    print(f"Step 1: Checking file...")
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        exit(1)
    print(f"  File exists: {input_file}")
    
    print(f"\nStep 2: Loading data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} companies")
    
    print(f"\nStep 3: Calculating scores...")
    df_scored = calculate_investment_score(df, verbose=True)
    
    print(f"\nStep 4: Ranking...")
    df_ranked = rank_startups(df_scored)
    
    print(f"\nStep 5: Extracting top 100...")
    top_100 = get_top_startups(df_ranked, n=100)
    print(f"  Got top 100 startups")
    
    # Display Top 10
    print(f"\n{'='*100}")
    print("TOP 10 STARTUPS")
    print(f"{'='*100}")
    cols = ['rank', 'company', 'stage', 'investment_score', 'rule_of_40', 'traction_index', 'burn_multiple', 'funding_amount']
    print(top_100[cols].head(10).to_string(index=False))
    
    # Summary statistics
    scoring_summary(df_ranked)
    
    # Detailed breakdown for #1
    if len(top_100) > 0:
        top_company = top_100.iloc[0]['company']
        print(f"\nDetailed breakdown for #{1}:")
        score_breakdown(df_ranked, top_company)
    
    # Save results
    print(f"\nStep 6: Saving results...")
    output_file = Path("data/processed/startups_scored.csv")
    df_ranked.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    top_100_file = Path("data/processed/top_100_startups.csv")
    top_100.to_csv(top_100_file, index=False)
    print(f"  Saved: {top_100_file}")
    
    print("\n=== SCRIPT COMPLETED SUCCESSFULLY ===")