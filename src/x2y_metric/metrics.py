import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def x2y(x, y):
    """Calculate X2Y metric for any x and y."""
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

    is_y_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
    if is_y_categorical:
        baseline_pred = y.mode()[0]
        baseline_error = 1 - (y == baseline_pred).mean()
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        error_metric = lambda y_true, y_pred: 1 - (y_true == y_pred).mean()
    else:
        baseline_pred = y.mean()
        baseline_error = mean_absolute_error(y, [baseline_pred] * len(y))
        model = DecisionTreeRegressor(random_state=42, max_depth=3)
        error_metric = mean_absolute_error

    model.fit(X, y)
    preds = model.predict(X)
    model_error = error_metric(y, preds)

    if baseline_error == 0:
        return 0.0 if model_error == 0 else 100.0
    reduction = (baseline_error - model_error) / baseline_error
    return max(0.0, min(100.0, reduction * 100))