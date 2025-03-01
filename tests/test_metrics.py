"""Tests for x2y-metric functionality."""

import numpy as np
import pandas as pd

from x2y_metric.metrics import dx2y, x2y


def test_continuous_x_continuous_y_linear():
    """Test X2Y for a perfect linear relationship."""
    x = np.array([1, 2, 3, 4])
    y = np.array([2, 4, 6, 8])
    result = x2y(x, y)
    assert result == 100.0, f"Expected 100%, got {result}"

def test_semicircle():
    """Test X2Y for a nonlinear semicircle relationship."""
    np.random.seed(42)
    x = np.arange(-1, 1.01, 0.01)
    y = np.sqrt(1 - x**2) + np.random.normal(0, 0.05, len(x))
    result = x2y(x, y)
    assert 60 < result < 80, f"Expected ~68%, got {result}"

def test_categorical_x_continuous_y():
    """Test X2Y for categorical x and continuous y."""
    x = pd.Series(["A", "B", "A", "B"])
    y = pd.Series([1.0, 2.0, 1.1, 1.9])
    result = x2y(x, y)
    assert 70 < result < 90, f"Expected 70-90%, got {result}"

def test_continuous_x_categorical_y():
    """Test X2Y for continuous x and categorical y."""
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series(["A", "B", "A", "B"])
    result = x2y(x, y)
    assert result == 100.0, f"Expected 100%, got {result}"

def test_categorical_x_categorical_y():
    """Test X2Y for categorical x and y."""
    x = pd.Series(["A", "B", "A", "B"])
    y = pd.Series(["X", "Y", "X", "Y"])
    result = x2y(x, y)
    assert result == 100.0, f"Expected 100%, got {result}"

def test_dx2y():
    """Test dx2y for pairwise X2Y calculation."""
    data = pd.DataFrame({
        "cat_x": ["A", "B", "A", "B"],
        "cont_y": [1.0, 2.0, 1.1, 1.9],
        "cat_z": ["X", "Y", "X", "Y"]
    })
    result = dx2y(data)
    assert result.shape == (3, 3)
    assert result["cat_x"]["cat_x"] == 100.0
