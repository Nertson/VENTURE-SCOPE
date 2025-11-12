import pandas as pd

# Charger les données enrichies
df = pd.read_csv('data/processed/startups_enriched.csv')

print("📊 Analyse des Données Manquantes pour investors_count\n")
print("=" * 60)

# Total
total = len(df)
missing = df['investors_count'].isna().sum()
present = total - missing

print(f"Total companies: {total:,}")
print(f"With investors_count: {present:,} ({present/total*100:.1f}%)")
print(f"Missing investors_count: {missing:,} ({missing/total*100:.1f}%)")

# Par stage
print("\n📈 Missing Rate by Stage:")
print("-" * 60)
stage_missing = df.groupby('stage', dropna=False).agg({
    'company': 'count',
    'investors_count': lambda x: x.isna().sum()
})
stage_missing['missing_%'] = (stage_missing['investors_count'] / stage_missing['company'] * 100).round(1)
stage_missing = stage_missing.sort_values('missing_%', ascending=False)
print(stage_missing)

# Par pays (Top 10)
print("\n🌍 Missing Rate by Country (Top 10 countries):")
print("-" * 60)
country_analysis = df.groupby('country', dropna=False).agg({
    'company': 'count',
    'investors_count': lambda x: x.isna().sum()
})
country_analysis['missing_%'] = (country_analysis['investors_count'] / country_analysis['company'] * 100).round(1)
country_top = country_analysis[country_analysis['company'] >= 50].sort_values('missing_%', ascending=False).head(10)
print(country_top)

# Par montant de funding
print("\n💰 Missing Rate by Funding Amount:")
print("-" * 60)
df['funding_bucket'] = pd.cut(df['funding_amount'], 
                               bins=[0, 100000, 1000000, 10000000, 100000000, float('inf')],
                               labels=['<100K', '100K-1M', '1M-10M', '10M-100M', '>100M'])
funding_analysis = df.groupby('funding_bucket', observed=True).agg({
    'company': 'count',
    'investors_count': lambda x: x.isna().sum()
})
funding_analysis['missing_%'] = (funding_analysis['investors_count'] / funding_analysis['company'] * 100).round(1)
print(funding_analysis)

# Exemples de companies avec funding mais sans investors
print("\n🔍 Sample: Companies with funding but NO investors_count:")
print("-" * 60)
no_investors = df[df['investors_count'].isna()].head(10)
print(no_investors[['company', 'stage', 'country', 'funding_amount']].to_string())

# Distribution des investor counts (pour ceux qui existent)
print("\n📊 Investor Count Distribution (for available data):")
print("-" * 60)
print(df['investors_count'].describe())

# Corrélation funding vs investors
print("\n🔗 Correlation: Funding Amount vs Investor Count:")
print("-" * 60)
correlation = df[['funding_amount', 'investors_count']].corr().iloc[0, 1]
print(f"Pearson correlation: {correlation:.3f}")