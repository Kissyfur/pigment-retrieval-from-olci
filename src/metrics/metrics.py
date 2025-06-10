import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)
from src.models.my_models.Annalisa import Annalisa

annalisa = Annalisa()
logging.basicConfig(level=logging.INFO)


def drop_nans(ty, py):
    t_not_nans = ~np.isnan(ty)
    p_not_nans = ~np.isnan(py)
    not_nans = t_not_nans * p_not_nans
    ty = ty[not_nans]
    py = py[not_nans]
    return ty, py


def exponential_r2(y, py):
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return -999
    y = np.exp(y)
    py = np.exp(py)
    return r2_score(y, py)

# return list(mean_absolute_error(y, py, multioutput='raw_values'))

def r2(y, py):
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        return -999
    return r2_score(y, py)


def exponential_mape(y, py):
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        return 500
    y = np.exp(y)
    py = np.exp(py)
    return mean_absolute_percentage_error(y, py)


def exponential_mae(y, py):
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return 500
    y = np.exp(y)
    py = np.exp(py)
    return mean_absolute_error(y, py)


def exponential_mse(y, py):
    if np.any(np.isnan(py)) or np.any(np.isinf(py)):
        print("NaN in py")
        return 500
    y = np.exp(y)
    py = np.exp(py)
    return mean_squared_error(y, py)


def mae_mse_loss(y, py):
    mean_absolute_error(y, py) + mean_squared_error(y, py)


class Metrics:
    M_FUNCTIONS = {"R2": r2_score,
                   "MAPE": exponential_mape,
                   "MAE": exponential_mae,
                   "MSE": exponential_mse,
                   }
    MET_NAMES = M_FUNCTIONS.keys()

    def __init__(self, met_names=MET_NAMES):
        self.metrics = {m_name:  self.M_FUNCTIONS[m_name] for m_name in met_names}

    def compute_metrics_df(self, ty, py):
        m = self.compute_metrics(ty.values, py.values)
        return pd.DataFrame(m, index=ty.columns).T

    def compute_metrics(self, ty, py):
        cols = ty.shape[1]
        mets = []
        for col in range(cols):
            t, p = drop_nans(ty[:, col], py[:, col])
            if len(t) == 0:
                logging.info(f"Can not compute metrics on column {col} due to NaN's")
                mets.append([np.nan for met in self.metrics])
                continue
            mets.append({met_name: met(t, p) for met_name, met in self.metrics.items()})
        return mets

def my_r2_score(ty, py):
    dim_y = ty.shape[1]

    for i in range(dim_y):
        t, p = 6
    r2_score

