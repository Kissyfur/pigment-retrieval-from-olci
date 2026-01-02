import pandas as pd
import argparse
import logging
import pickle
import numpy as np

from pathlib import Path
from src.models.keras_models import ConcatenatedModulesModel
from src.models.sklearn_models import RandomForestModel, XGBModel

logging.basicConfig(level=logging.INFO)
output_labels = ["chlid", "chl_a", "chl_b", "chc12", "fucox", "hxfcx", "btfcx", "diadi", "allox", "diato", "zeaxa",
                   "betac", "perid"]
input_labels = ["400", "412", "442", "490", "510", "560", "620", "665", "673", "681", "708", "778", "865"]

dir_predictions = Path('data/predictions')
dir_predictions.mkdir(parents=True, exist_ok=True)


class_instance = {'rf': RandomForestModel('rf'),
                  'xgb': XGBModel('xgb'),
                  'cnn': ConcatenatedModulesModel('concatenatedCNN'),
                  'dnn': ConcatenatedModulesModel('concatenatedDNN'),
                  'bilstm': ConcatenatedModulesModel('concatenatedBiLSTM')
                  }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict pigments from radiometry')
    parser.add_argument('--path_data', metavar='p', type=str, help='path to data', required=True)
    parser.add_argument('--fn_save', metavar='f', type=str,
                        help='filename for saving file with predictions (.csv extension)', default='tmp')
    parser.add_argument('--model', default='cnn', const='xgb', nargs='?',
                        choices=['xgb', 'rf', 'random_forest', 'xgboost', 'bilstm', 'dnn', 'cnn'])
    parser.add_argument('--exp_name', metavar='l', type=str, help='experiment name',
                        default='experiments/13_wl_final')

    args = parser.parse_args()
    path_data = args.path_data
    fn = args.fn_save + '.csv'
    mn = args.model
    en = args.exp_name

    models = {}
    scalers = {}
    data = pd.read_csv(path_data, index_col=0)[input_labels]
    n = len(list(Path(en).iterdir()))
    pys = {}

    for split in range(n):
        with open(f'{en}/split_{split}/scaler_y.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        scalers[split] = loaded_scaler
        path_model = f'{en}/split_{split}/models'
        mod_ = class_instance[mn]
        mod_.load(path_model)
        py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(data)), columns=output_labels,
                          index=data.index)
        pys[split] = py
        py.to_csv(dir_predictions/f'{mod_.name}_split_{split}_{fn}')

    final_py = np.mean([py_ for py_ in pys.values()], axis=0)
    final_py = pd.DataFrame(final_py, columns=output_labels, index=data.index)
    final_py.to_csv(dir_predictions/f'{mod_.name}_{fn}')
