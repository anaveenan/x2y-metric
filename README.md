# x2y-metric

[![PyPI version](https://badge.fury.io/py/x2y-metric.svg)](https://badge.fury.io/py/x2y-metric)
[![Test and Lint](https://github.com/anaveenan/x2y-metric/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/anaveenan/x2y-metric/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/x2y-metric.svg)](https://pypi.org/project/x2y-metric/)

Compute the X2Y metric for detecting relationships between variables, inspired by Dr. Rama Ramakrishnan’s approach. This package uses decision trees to assess linear and nonlinear relationships across continuous and categorical variables.

## Installation
```bash
pip install x2y-metric

## Installation
```bash
pip install x2y-metric


## Usage
```python
from x2y_metric import x2y, dx2y
import pandas as pd

# Single pair
x = [1, 2, 3]
y = [2, 4, 6]
print(x2y(x, y))  # ~100%

# DataFrame
df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
print(dx2y(df))
```

Build:
```bash
uv build
