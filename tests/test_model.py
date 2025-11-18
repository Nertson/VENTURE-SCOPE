"""
Basic tests for VENTURE-SCOPE ML model.
"""

import pytest
import pandas as pd
from pathlib import Path


def test_model_exists():
    """Test that trained model file exists."""
    model_path = Path("results/models/random_forest.pkl")
    assert model_path.exists(), "Trained model not found"


def test_data_exists():
    """Test that processed data exists."""
    data_path = Path("data/processed/startups_scored.csv")
    assert data_path.exists(), "Processed data not found"


def test_data_loading():
    """Test that data can be loaded."""
    df = pd.read_csv("data/processed/startups_scored.csv")
    assert len(df) > 0, "Data is empty"
    assert 'investment_score' in df.columns, "Missing investment_score column"


def test_results_exist():
    """Test that results files exist."""
    assert Path("results/model_comparison.csv").exists()
    assert Path("results/top_100_startups.csv").exists()


def test_visualizations_exist():
    """Test that visualizations were generated."""
    figures_dir = Path("results/figures")
    assert figures_dir.exists()
    assert (figures_dir / "model_comparison.png").exists()
    assert (figures_dir / "confusion_matrix.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])