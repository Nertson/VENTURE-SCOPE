from __future__ import annotations
from pathlib import Path
import pandas as pd

CANONICAL_COLS = (
    "company","stage","country","sector",
    "funding_amount","investors_count","founded_year"
)

STAGE_MAP = {
    "pre-seed":"Pre-Seed","seed":"Seed","angel":"Angel",
    "series a":"Series A","series b":"Series B","series c":"Series C",
    "series d":"Series D+","series e":"Series D+"
}

def _standardize_stage(x):
    if not isinstance(x,str) or not x.strip(): return None
    return STAGE_MAP.get(x.lower().strip(), x.strip().title())

def _coalesce(df: pd.DataFrame, cands: list[str], new_col: str) -> pd.Series:
    s = pd.Series([None]*len(df))
    for c in cands:
        if c in df.columns: s = s.fillna(df[c])
    return s.rename(new_col)

def load_startups_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "company" not in df.columns:
        for cand in ("company","organization","startup","name"):
            if cand in df.columns:
                df = df.rename(columns={cand:"company"})
                break
    if "stage" not in df.columns: df["stage"] = None
    df["stage"] = df["stage"].apply(_standardize_stage)
    if "sector" not in df.columns:  df["sector"]  = _coalesce(df, ["sector","category","industry"], "sector")
    if "country" not in df.columns: df["country"] = _coalesce(df, ["country","country_code","hq_country"], "country")
    for col in ("funding_amount","investors_count","founded_year"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for c in CANONICAL_COLS:
        if c not in df.columns: df[c] = pd.NA
    out = df[list(CANONICAL_COLS)].copy()
    for c in ("company","stage","country","sector"):
        out[c] = out[c].astype("string").str.strip()
    return out
