#!/usr/bin/env python3
"""
Interactive Startup Success Predictor for VENTURE-SCOPE

This script allows users to input startup characteristics and receive
an ML-powered prediction of success probability based on historical patterns.

Usage:
    python src/venture_scope/ml/predict.py
    
Or interactively:
    from venture_scope.ml.predict import predict_startup
    predict_startup(funding=10000000, stage='Series A', ...)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("🚀 VENTURE-SCOPE: Startup Success Predictor")
print("=" * 70)


# ==================== CONFIGURATION ====================

STAGES = ['Seed', 'Angel', 'Series A', 'Series B', 'Series C', 'Series D+']
COMMON_SECTORS = ['saas', 'web', 'mobile', 'biotech', 'fintech', 'ecommerce', 
                  'enterprise', 'cleantech', 'hardware', 'other']
COMMON_COUNTRIES = ['USA', 'GBR', 'CHN', 'CAN', 'DEU', 'FRA', 'IND', 'ISR', 'other']

# Stage-based defaults for estimation
STAGE_DEFAULTS = {
    'Seed': {
        'burn_period': 18,
        'revenue_multiple': 0.10,
        'rule_40_base': 100,
        'stage_weight': 1.0
    },
    'Angel': {
        'burn_period': 18,
        'revenue_multiple': 0.08,
        'rule_40_base': 90,
        'stage_weight': 1.0
    },
    'Series A': {
        'burn_period': 24,
        'revenue_multiple': 0.30,
        'rule_40_base': 100,
        'stage_weight': 1.5
    },
    'Series B': {
        'burn_period': 30,
        'revenue_multiple': 0.50,
        'rule_40_base': 80,
        'stage_weight': 2.0
    },
    'Series C': {
        'burn_period': 36,
        'revenue_multiple': 0.80,
        'rule_40_base': 50,
        'stage_weight': 2.5
    },
    'Series D+': {
        'burn_period': 36,
        'revenue_multiple': 1.00,
        'rule_40_base': 40,
        'stage_weight': 3.0
    }
}


# ==================== KPI CALCULATION ====================

def calculate_kpis(
    funding_amount: float,
    stage: str,
    investors_count: int,
    founded_year: int,
    current_year: int = 2025
) -> Dict[str, float]:
    """
    Calculate KPIs for a startup based on basic inputs.
    
    Args:
        funding_amount: Total funding raised ($)
        stage: Funding stage (Seed, Series A, etc.)
        investors_count: Number of unique investors
        founded_year: Year company was founded
        current_year: Current year for age calculation
    
    Returns:
        Dictionary with all calculated KPIs
    """
    defaults = STAGE_DEFAULTS.get(stage, STAGE_DEFAULTS['Series A'])
    
    # Company age
    age = max(1, current_year - founded_year)
    
    # Estimated Revenue
    estimated_revenue = funding_amount * defaults['revenue_multiple']
    
    # Capital Efficiency
    capital_efficiency = estimated_revenue / funding_amount if funding_amount > 0 else 0
    capital_efficiency = min(1.0, capital_efficiency)  # Cap at 1.0
    
    # Monthly Burn
    monthly_burn = funding_amount / defaults['burn_period']
    
    # Runway (assume 50% of funding still available)
    available_cash = funding_amount * 0.5
    runway_months = available_cash / monthly_burn if monthly_burn > 0 else 0
    runway_months = min(24, runway_months)  # Cap at 24 months
    
    # Burn Multiple
    annual_burn = monthly_burn * 12
    burn_multiple = annual_burn / estimated_revenue if estimated_revenue > 0 else 10
    burn_multiple = min(10, max(0.3, burn_multiple))  # Clip between 0.3 and 10
    
    # Traction Index
    funding_log = np.log10(max(100000, funding_amount))  # Min $100K
    traction_raw = (funding_log * investors_count * defaults['stage_weight']) / age
    traction_index = min(100, traction_raw)  # Scale to 0-100
    
    # Rule of 40 (estimated)
    rule_40_adjustment = (capital_efficiency - 0.30) * 50
    rule_of_40 = defaults['rule_40_base'] + rule_40_adjustment
    rule_of_40 = max(0, min(150, rule_of_40))  # Clip 0-150
    
    # Investment Score (weighted combination)
    rule_40_norm = min(100, rule_of_40)
    traction_norm = traction_index
    cap_eff_norm = capital_efficiency * 100
    burn_norm = (1 / burn_multiple) * 50 if burn_multiple > 0 else 0
    burn_norm = min(100, burn_norm)
    runway_norm = (runway_months / 24) * 100
    
    investment_score = (
        rule_40_norm * 0.25 +
        traction_norm * 0.25 +
        cap_eff_norm * 0.20 +
        burn_norm * 0.15 +
        runway_norm * 0.15
    )
    
    return {
        'estimated_revenue': estimated_revenue,
        'capital_efficiency': capital_efficiency,
        'monthly_burn': monthly_burn,
        'runway_months': runway_months,
        'burn_multiple': burn_multiple,
        'traction_index': traction_index,
        'rule_of_40': rule_of_40,
        'investment_score': investment_score,
        'age': age
    }


# ==================== FEATURE ENGINEERING ====================

def prepare_features(
    funding_amount: float,
    stage: str,
    sector: str,
    country: str,
    investors_count: int,
    founded_year: int,
    kpis: Dict[str, float],
    model
) -> pd.DataFrame:
    """
    Prepare features in the format expected by the trained model.
    
    Creates a DataFrame with ALL columns the model expects (113 total).
    """
    # Get feature names from the trained model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
    else:
        # Fallback: manually create expected feature list
        # This should match what was used during training
        expected_features = []
        
        # Numeric features (8)
        numeric_features = [
            'funding_amount', 'investors_count', 'rule_of_40', 
            'traction_index', 'capital_efficiency', 'burn_multiple',
            'runway_months', 'investment_score'
        ]
        expected_features.extend(numeric_features)
        
        # Stage features (one-hot encoded)
        for stage_val in ['Seed', 'Angel', 'Series A', 'Series B', 'Series C', 'Series D+']:
            expected_features.append(f'stage_{stage_val}')
        
        # We'll add sector and country dynamically based on what model has
        print("⚠️  Warning: Could not get feature names from model, using fallback")
    
    # Create a dictionary with ALL features initialized to 0
    features = {feat: 0 for feat in expected_features}
    
    # Fill in the numeric features
    features['funding_amount'] = funding_amount
    features['investors_count'] = investors_count
    features['rule_of_40'] = kpis['rule_of_40']
    features['traction_index'] = kpis['traction_index']
    features['capital_efficiency'] = kpis['capital_efficiency']
    features['burn_multiple'] = kpis['burn_multiple']
    features['runway_months'] = kpis['runway_months']
    features['investment_score'] = kpis['investment_score']
    
    # Fill in stage (one-hot encoding)
    stage_col = f'stage_{stage}'
    if stage_col in features:
        features[stage_col] = 1
    
    # Fill in sector (one-hot encoding)
    sector_lower = sector.lower()
    sector_col = f'sector_{sector_lower}'
    if sector_col in features:
        features[sector_col] = 1
    
    # Fill in country (one-hot encoding)
    country_upper = country.upper()
    country_col = f'country_{country_upper}'
    if country_col in features:
        features[country_col] = 1
    
    # Convert to DataFrame with columns in the EXACT order expected by model
    df = pd.DataFrame([features], columns=expected_features)
    
    return df


# ==================== MODEL LOADING ====================

def load_model(model_path: str = "results/models/random_forest.pkl") -> Optional[object]:
    """Load the trained Random Forest model."""
    try:
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"❌ Model not found at {model_path}")
            print(f"   Please run: python src/venture_scope/ml/model.py")
            return None
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# ==================== PREDICTION ====================

def predict_success(
    model,
    features: pd.DataFrame
) -> Tuple[float, str]:
    """
    Predict success probability for a startup.
    
    Returns:
        (probability, confidence_level)
    """
    # Get prediction probability
    prob = model.predict_proba(features)[0]
    success_prob = prob[1]  # Probability of success (class 1)
    
    # Determine confidence level based on probability
    if success_prob > 0.8 or success_prob < 0.2:
        confidence = "HIGH"
    elif success_prob > 0.65 or success_prob < 0.35:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return success_prob, confidence


# ==================== INTERPRETATION ====================

def interpret_prediction(
    success_prob: float,
    kpis: Dict[str, float],
    funding_amount: float,
    stage: str,
    investors_count: int
) -> Dict[str, list]:
    """
    Generate human-readable interpretation of prediction.
    
    Returns:
        Dictionary with 'strengths' and 'concerns' lists
    """
    strengths = []
    concerns = []
    
    # Funding
    if stage == 'Seed' and funding_amount > 2_000_000:
        strengths.append(f"Strong seed funding (${funding_amount/1e6:.1f}M)")
    elif stage == 'Series A' and funding_amount > 8_000_000:
        strengths.append(f"Strong Series A (${funding_amount/1e6:.1f}M)")
    elif stage == 'Series B' and funding_amount > 20_000_000:
        strengths.append(f"Strong Series B (${funding_amount/1e6:.1f}M)")
    
    # Investors
    if investors_count >= 5:
        strengths.append(f"Good investor validation ({investors_count} investors)")
    elif investors_count <= 2:
        concerns.append(f"Limited investor validation ({investors_count} investors)")
    
    # Capital Efficiency
    if kpis['capital_efficiency'] > 0.40:
        strengths.append(f"Strong capital efficiency ({kpis['capital_efficiency']:.2f})")
    elif kpis['capital_efficiency'] < 0.20:
        concerns.append(f"Low capital efficiency ({kpis['capital_efficiency']:.2f})")
    
    # Burn Multiple
    if kpis['burn_multiple'] < 1.5:
        strengths.append(f"Efficient burn rate (${kpis['burn_multiple']:.1f} burned per $1 revenue)")
    elif kpis['burn_multiple'] > 3.0:
        concerns.append(f"High burn rate (${kpis['burn_multiple']:.1f} burned per $1 revenue)")
    
    # Runway
    if kpis['runway_months'] > 15:
        strengths.append(f"Healthy runway ({kpis['runway_months']:.0f} months)")
    elif kpis['runway_months'] < 9:
        concerns.append(f"Limited runway ({kpis['runway_months']:.0f} months)")
    
    # Traction
    if kpis['traction_index'] > 60:
        strengths.append(f"Strong traction index ({kpis['traction_index']:.0f}/100)")
    elif kpis['traction_index'] < 30:
        concerns.append(f"Low traction index ({kpis['traction_index']:.0f}/100)")
    
    # Investment Score
    if kpis['investment_score'] > 70:
        strengths.append(f"High investment score ({kpis['investment_score']:.0f}/100)")
    elif kpis['investment_score'] < 40:
        concerns.append(f"Below-average investment score ({kpis['investment_score']:.0f}/100)")
    
    return {'strengths': strengths, 'concerns': concerns}


def get_recommendation(success_prob: float) -> str:
    """Get investment recommendation based on probability."""
    if success_prob >= 0.75:
        return "🟢 STRONG INVEST - High success probability"
    elif success_prob >= 0.60:
        return "🟡 CONSIDER - Above average potential"
    elif success_prob >= 0.45:
        return "🟠 CAUTIOUS - Average risk/reward"
    else:
        return "🔴 PASS - Below average probability"


# ==================== USER INTERACTION ====================

def get_user_input() -> Dict:
    """Interactively collect startup information from user."""
    print("\n📋 Enter startup information:")
    print("-" * 70)
    
    # Funding amount
    while True:
        try:
            funding_str = input("  Funding raised (e.g., 10000000 for $10M): $")
            funding_amount = float(funding_str)
            if funding_amount <= 0:
                print("     ⚠️  Funding must be positive")
                continue
            break
        except ValueError:
            print("     ⚠️  Please enter a valid number")
    
    # Stage
    print(f"\n  Available stages: {', '.join(STAGES)}")
    while True:
        stage = input("  Stage: ").strip()
        if stage in STAGES:
            break
        print(f"     ⚠️  Please choose from: {', '.join(STAGES)}")
    
    # Sector
    print(f"\n  Common sectors: {', '.join(COMMON_SECTORS)}")
    sector = input("  Sector: ").strip().lower()
    if not sector:
        sector = 'other'
    
    # Country
    print(f"\n  Common countries: {', '.join(COMMON_COUNTRIES)}")
    country = input("  Country (e.g., USA, GBR): ").strip().upper()
    if not country:
        country = 'USA'
    
    # Investors
    while True:
        try:
            investors_count = int(input("  Number of investors: "))
            if investors_count < 0:
                print("     ⚠️  Cannot be negative")
                continue
            break
        except ValueError:
            print("     ⚠️  Please enter a valid number")
    
    # Founded year
    while True:
        try:
            founded_year = int(input("  Founded year (e.g., 2020): "))
            if founded_year < 1990 or founded_year > 2025:
                print("     ⚠️  Please enter a realistic year (1990-2025)")
                continue
            break
        except ValueError:
            print("     ⚠️  Please enter a valid year")
    
    return {
        'funding_amount': funding_amount,
        'stage': stage,
        'sector': sector,
        'country': country,
        'investors_count': investors_count,
        'founded_year': founded_year
    }


# ==================== DISPLAY RESULTS ====================

def display_results(
    inputs: Dict,
    kpis: Dict,
    success_prob: float,
    confidence: str,
    interpretation: Dict
):
    """Display prediction results in a beautiful format."""
    
    print("\n" + "=" * 70)
    print("📊 CALCULATED KPIs")
    print("=" * 70)
    print(f"  Estimated Revenue:     ${kpis['estimated_revenue']:,.0f}/year")
    print(f"  Capital Efficiency:    {kpis['capital_efficiency']:.2f} ({kpis['capital_efficiency']*100:.0f}%)")
    print(f"  Monthly Burn:          ${kpis['monthly_burn']:,.0f}/month")
    print(f"  Runway:                {kpis['runway_months']:.0f} months")
    print(f"  Burn Multiple:         {kpis['burn_multiple']:.2f}x")
    print(f"  Traction Index:        {kpis['traction_index']:.0f}/100")
    print(f"  Rule of 40:            {kpis['rule_of_40']:.0f}")
    print(f"  Investment Score:      {kpis['investment_score']:.0f}/100")
    
    print("\n" + "=" * 70)
    print("🔮 PREDICTION")
    print("=" * 70)
    print(f"  Success Probability:   {success_prob*100:.1f}%")
    print(f"  Confidence:            {confidence}")
    print(f"  Recommendation:        {get_recommendation(success_prob)}")
    
    print("\n" + "=" * 70)
    print("💡 INTERPRETATION")
    print("=" * 70)
    
    # Contextualization
    if success_prob >= 0.80:
        percentile = "top 5%"
    elif success_prob >= 0.70:
        percentile = "top 15%"
    elif success_prob >= 0.60:
        percentile = "top 30%"
    elif success_prob >= 0.50:
        percentile = "above median"
    else:
        percentile = "below median"
    
    print(f"\n  This startup has characteristics similar to the {percentile}")
    print(f"  of successful VC-backed companies in our historical dataset.")
    
    # Strengths
    if interpretation['strengths']:
        print(f"\n  ✅ Key Strengths:")
        for strength in interpretation['strengths']:
            print(f"     • {strength}")
    
    # Concerns
    if interpretation['concerns']:
        print(f"\n  ⚠️  Areas to Watch:")
        for concern in interpretation['concerns']:
            print(f"     • {concern}")
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("""
  This prediction is based on patterns from 2013 historical data.
  
  ✅ Appropriate use:
     - Benchmarking against historical successful companies
     - Relative comparison between startups
     - Identifying key strengths and concerns
  
  ❌ Limitations:
     - Market conditions have changed since 2013
     - Some metrics are estimated (not actual financials)
     - NOT investment advice - use for informational purposes only
  
  For production decisions, combine with:
     - Human due diligence
     - Current market research
     - Actual financial statements
    """)
    print("=" * 70)


# ==================== MAIN FUNCTION ====================

def predict_startup(
    funding_amount: Optional[float] = None,
    stage: Optional[str] = None,
    sector: Optional[str] = None,
    country: Optional[str] = None,
    investors_count: Optional[int] = None,
    founded_year: Optional[int] = None
) -> Optional[Dict]:
    """
    Predict startup success probability.
    
    Can be called interactively (no args) or programmatically (with args).
    
    Returns:
        Dictionary with prediction results or None if error
    """
    # Load model
    model = load_model()
    if model is None:
        return None
    
    # Get inputs (interactive or programmatic)
    if funding_amount is None:
        inputs = get_user_input()
    else:
        inputs = {
            'funding_amount': funding_amount,
            'stage': stage or 'Series A',
            'sector': sector or 'saas',
            'country': country or 'USA',
            'investors_count': investors_count or 3,
            'founded_year': founded_year or 2020
        }
    
    print("\n⏳ Calculating KPIs...")
    
    # Calculate KPIs
    kpis = calculate_kpis(
        funding_amount=inputs['funding_amount'],
        stage=inputs['stage'],
        investors_count=inputs['investors_count'],
        founded_year=inputs['founded_year']
    )
    
    print("✅ KPIs calculated")
    print("⏳ Preparing features...")
    
    # Prepare features for model
    features = prepare_features(
        funding_amount=inputs['funding_amount'],
        stage=inputs['stage'],
        sector=inputs['sector'],
        country=inputs['country'],
        investors_count=inputs['investors_count'],
        founded_year=inputs['founded_year'],
        kpis=kpis,
        model=model
    )
    
    print("✅ Features prepared")
    print("⏳ Running prediction...")
    
    # Predict
    try:
        success_prob, confidence = predict_success(model, features)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None
    
    print("✅ Prediction complete")
    
    # Interpret
    interpretation = interpret_prediction(
        success_prob=success_prob,
        kpis=kpis,
        funding_amount=inputs['funding_amount'],
        stage=inputs['stage'],
        investors_count=inputs['investors_count']
    )
    
    # Display results
    display_results(inputs, kpis, success_prob, confidence, interpretation)
    
    # Return results for programmatic use
    return {
        'inputs': inputs,
        'kpis': kpis,
        'success_probability': success_prob,
        'confidence': confidence,
        'interpretation': interpretation
    }


# ==================== CLI ENTRY POINT ====================

def main():
    """Command-line interface entry point."""
    print("\nWelcome to the VENTURE-SCOPE Startup Success Predictor!")
    print("\nThis tool predicts startup success probability based on")
    print("machine learning trained on historical VC-backed companies.\n")
    
    while True:
        result = predict_startup()
        
        if result is None:
            break
        
        print("\n" + "=" * 70)
        again = input("\n🔄 Predict another startup? (y/n): ").strip().lower()
        if again != 'y':
            break
        print("\n" + "=" * 70 + "\n")
    
    print("\n👋 Thank you for using VENTURE-SCOPE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()