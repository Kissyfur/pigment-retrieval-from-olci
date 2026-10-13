import numpy as np

from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)


def exponential_r2_per_class(y, py):
    if len(y.shape) != 2:
        return 500
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return [-999] * y.shape[1]
    y = np.exp(y)
    py = np.exp(py)
    return list(r2_score(y, py, multioutput='raw_values'))


def r2_per_class(y, py):
    if len(y.shape) != 2:
        return -999
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return [-999] * y.shape[1]
    return list(r2_score(y, py, multioutput='raw_values'))


def exponential_mape_per_class(y, py):
    if len(y.shape) != 2:
        return 500
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return [500] * y.shape[1]
    y = np.exp(y)
    py = np.exp(py)
    return list(mean_absolute_percentage_error(y, py, multioutput='raw_values'))


def exponential_mae_per_class(y, py):
    if len(y.shape) != 2:
        return 500
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return [500] * y.shape[1]
    y = np.exp(y)
    py = np.exp(py)
    return list(mean_absolute_error(y, py, multioutput='raw_values'))


def exponential_mse_per_class(y, py):
    if len(y.shape) != 2:
        return 500
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return [500] * y.shape[1]
    y = np.exp(y)
    py = np.exp(py)
    return list(mean_squared_error(y, py, multioutput='raw_values'))