# x2y-metric
Compute the X2Y metric for detecting relationships between variables.

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
