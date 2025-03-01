"""Module for computing the X2Y metric to detect variable relationships."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def x2y(x, y):
    """Calculate X2Y metric for any x and y."""
    x = pd.Series(x)
    y = pd.Series(y)
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return 0.0

    is_x_categorical = (pd.api.types.is_categorical_dtype(x) or
                        pd.api.types.is_object_dtype(x))
    if is_x_categorical:
        le = LabelEncoder()
        x_encoded = le.fit_transform(x).reshape(-1, 1)
    else:
        x_encoded = x.values.reshape(-1, 1)

    is_y_categorical = (pd.api.types.is_categorical_dtype(y) or
                        pd.api.types.is_object_dtype(y))
    if is_y_categorical:
        baseline_pred = y.mode()[0]
        baseline_error = 1 - (y == baseline_pred).mean()
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        def misclassification_error(y_true, y_pred):
            return 1 - (y_true == y_pred).mean()
        error_metric = misclassification_error
    else:
        baseline_pred = y.mean()
        baseline_error = mean_absolute_error(y, [baseline_pred] * len(y))
        model = DecisionTreeRegressor(random_state=42, max_depth=3)
        error_metric = mean_absolute_error

    model.fit(x_encoded, y)
    preds = model.predict(x_encoded)
    model_error = error_metric(y, preds)

    if baseline_error == 0:
        return 0.0 if model_error == 0 else 100.0
    reduction = (baseline_error - model_error) / baseline_error
    return max(0.0, min(100.0, reduction * 100))

def dx2y(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate X2Y metric for all pairs in a DataFrame."""
    cols = data.columns
    n = len(cols)
    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                forward = x2y(data[cols[i]], data[cols[j]])
                reverse = x2y(data[cols[j]], data[cols[i]])
                result[i, j] = result[j, i] = (forward + reverse) / 2

    return pd.DataFrame(result, index=cols, columns=cols)
