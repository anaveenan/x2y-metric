import numpy as np
from x2y_metric.metrics import x2y

def test_continuous_x_continuous_y_linear():
    x = np.array([1, 2, 3, 4])
    y = np.array([2, 4, 6, 8])
    result = x2y(x, y)
    assert result == 100.0, f"Expected 100%, got {result}"


def test_semicircle():
    np.random.seed(42)
    x = np.arange(-1, 1.01, 0.01)
    y = np.sqrt(1 - x**2) + np.random.normal(0, 0.05, len(x))
    result = x2y(x, y)
    assert 60 < result < 80, f"Expected ~68%, got {result}"