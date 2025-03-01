import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def x2y(x, y):
    """Calculate X2Y metric for any x and continuous y."""
    x = pd.Series(x)
    y = pd.Series(y)
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return 0.0

    is_x_categorical = pd.api.types.is_categorical_dtype(x) or pd.api.types.is_object_dtype(x)
    if is_x_categorical:
        le = LabelEncoder()
        X = le.fit_transform(x).reshape(-1, 1)
    else:
        X = x.values.reshape(-1, 1)

    baseline_pred = y.mean()
    baseline_error = mean_absolute_error(y, [baseline_pred] * len(y))

    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    model_error = mean_absolute_error(y, preds)

    if baseline_error == 0:
        return 0.0 if model_error == 0 else 100.0
    reduction = (baseline_error - model_error) / baseline_error
    return max(0.0, min(100.0, reduction * 100))