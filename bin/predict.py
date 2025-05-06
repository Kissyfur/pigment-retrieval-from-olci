import pandas as pd
import argparse
import logging
import pickle
import numpy as np

from pathlib import Path


logging.basicConfig(level=logging.INFO)
output_labels = ['chlide_a[mg*m^3]', 'chla[mg*m^3]', 'chlb[mg*m^3]', 'chlc1+c2[mg*m^3]',
       'fucox[mg*m^3]', "19'hxfcx[mg*m^3]", "19'btfcx[mg*m^3]",
       'diadino[mg*m^3]', 'allox[mg*m^3]', 'diatox[mg*m^3]', 'zeaxan[mg*m^3]',
       'beta_car[mg*m^3]', 'peridinin[mg*m^3]']
dir_model       = Path('model')
dir_predictions = Path('data/predictions')
dir_predictions.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict pigments from radiometry')
    parser.add_argument('--path_data', metavar='p', type=str, help='path to data', required=True)
    parser.add_argument('--fn_save', metavar='f', type=str,
                        help='filename for saving file with predictions (.csv extension)', default='tmp')
    parser.add_argument('--model', default='xgb', const='xgb', nargs='?',
                        choices=['xgb', 'rf', 'random_forest', 'xgboost', 'dnn', 'dense_neural_network'])
    parser.add_argument('--legacy', metavar='l', type=bool, help='legacy input data', default=True)

    args       = parser.parse_args()
    path_data  = args.path_data
    fn_save    = args.fn_save + '.csv'
    model_name = args.model
    legacy     = args.legacy

    if path_data[-4:] != '.csv':
        logging.info(f'File {path_data} is not a ".csv" file')
        exit()
    data = pd.read_csv(path_data, index_col=0)

    if model_name in ["rf", "random_forest"]:
        model_path = dir_model / 'rf_legacy.pkl' if legacy else dir_model / 'rf.pkl'

    elif model_name in ["xgb", "xgboost"]:
        model_path = dir_model / 'xgb_legacy.pkl' if legacy else dir_model / 'xgb.pkl'

    elif model_name in ["dnn", "dense_neural_network"]:
        model_path = dir_model / 'dnn_legacy.pkl' if legacy else dir_model / 'dnn.pkl'

    else:
        logging.info(f'Model "{model_name}" not valid')
        exit()

    logging.info(f'Loading model in {model_path}   ...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logging.info(f'Predicting pigments   ...')
    py = np.exp(model.predict(data))

    df = pd.DataFrame(py, columns=output_labels)
    df['lat'] = data['lat']
    df['lon'] = data['lon']
    logging.info(f'Saving predictions in {dir_predictions/fn_save}   ...')
    df.to_csv(dir_predictions/fn_save)
