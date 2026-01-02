import numpy as np
import pickle
import pandas as pd

from pathlib import Path
from src.models.keras_models import ConcatenatedModulesModel
from src.models.sklearn_models import RandomForestModel, XGBModel

class_instance = {'rf': RandomForestModel('rf'),
                  'xgb': XGBModel('xgb'),
                  'cnn': ConcatenatedModulesModel('concatenatedCNN'),
                  'dnn': ConcatenatedModulesModel('concatenatedDNN'),
                  'bilstm': ConcatenatedModulesModel('concatenatedBiLSTM')
                  }

OUT_VARS_SHORT = ["chlid", "chl_a", "chl_b", "chc12", "fucox", "hxfcx", "btfcx", "diadi", "allox", "diato", "zeaxa",
                  "betac", "perid"]

def predict(experiment_name, x_df, model_name, verbose=0):
    models = {}
    scalers = {}
    n = len(list(Path(f'{experiment_name}').iterdir()))
    pys = {}

    for split in range(n):
        with open(f'{experiment_name}/split_{split}/scaler_y.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        scalers[split] = loaded_scaler
        path_model = f'{experiment_name}/split_{split}/models'
        mod_ = class_instance[model_name]
        mod_.load(path_model)
        py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(x_df, verbose=verbose)),
                          index=x_df.index, columns=OUT_VARS_SHORT)
        pys[split] = py
        # py.to_csv(dir_predictions / f'{mod_.name}_split_{split}_{fn}')

    final_py = np.mean([py_ for py_ in pys.values()], axis=0)
    final_py = pd.DataFrame(final_py, index=x_df.index, columns=OUT_VARS_SHORT)
    return final_py