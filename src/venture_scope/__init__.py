"""
Data ingestion module for Venture-Scope.

This module provides multiple loaders:
- load_startups_csv: Generic CSV loader (fast, flexible)
- load_enriched_startups: Crunchbase-specific loader with JOIN (complete data)
"""

from .loaders import load_startups_csv
from .loaders_enriched import load_enriched_startups

__all__ = [
    'load_startups_csv',           # Generic loader
    'load_enriched_startups',      # Enriched Crunchbase loader
]