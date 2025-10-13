import argparse
import json
import pandas as pd
import pickle
from tqdm import tqdm

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.data.data_spliting import create_splits, load_split
from src.models.model_training import train_modules, train_model
from src.metrics.metrics import Metrics
from src.models.keras_models import ConcatenatedModulesModel, ConvolutionalModel, DenseModel, BilstmModel
from src.models.sklearn_models import RandomForestModel, XGBModel
# wv_5 = ['412', '442', '490', '560', '673']
# wv_8 = ['400',                      '510',        '620', '665',        '681', '708', '778', '865']

class_instance = {'rf': RandomForestModel('rf'),
                  'xgb': XGBModel('xgb'),
                  'cnn': ConcatenatedModulesModel('concatenatedCNN'),
                  'dnn': ConcatenatedModulesModel('concatenatedDNN'),
                  'bilstm': ConcatenatedModulesModel('concatenatedBiLSTM')
                  }

modules_class_instance = {
    'cnn': ConvolutionalModel(),
    'dnn': DenseModel(),
    'bilstm': BilstmModel()
}

mets = Metrics()

with open('hyperparameter_space.json', 'r') as f:
    hs = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training and evaluation')
    parser.add_argument('--exp_config', metavar='-e', type=str, help='models for the experiments',
                        default='test_experiment')
    parser.add_argument('--steps', metavar='-s', nargs='+', default=["compute_metrics"],
                        help='Steps of the experiment. Options:\nsplit_data\ntrain_modules\ntrain_models\n'
                             'compute_metrics\nall')

    args = parser.parse_args()
    steps = args.steps
    with open(args.exp_config, 'r') as f:
        exp_config = json.load(f)

    exp_folder = f'experiments/{exp_config["EXP_NAME"]}'

    # Read data and preprocess
    x = pd.read_csv(exp_config["INP_PATH"])
    y = pd.read_csv(exp_config["OUT_PATH"])
    x = x[exp_config["INP_VARS"]]
    y = y[exp_config["OUT_VARS"]]
    y = y.rename(columns=dict(zip(exp_config["OUT_VARS"], exp_config["OUT_VARS_SHORT"])))

    # Create splits (not necessary to run if splits have been already created
    if 'split_data' in steps or 'all' in steps:
        create_splits(x=x, y=y, n_splits=exp_config["N"],
                      clusters=exp_config["CLUSTERS"], test_size=exp_config["TEST_SIZE"], r=exp_config["R"],
                      path_save_splits=exp_folder)

    for split in tqdm(range(exp_config["N"]), desc="Splits"):
        train_ids, test_ids = load_split(exp_folder, split)
        x_train, y_train = x.loc[train_ids].copy(), y.loc[train_ids].copy()
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)
        path_save_scaler = Path(f'{exp_folder}/split_{split}/')
        path_save_scaler.mkdir(parents=True, exist_ok=True)
        # Save the scaler
        with open(path_save_scaler/'scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)

    # Train modules for deep learning models. This takes a lot of computing time (~12 hours for experiment_1)
    if "train_modules" in steps or 'all' in steps:
        print(f"Training modules...")
        for split in tqdm(range(exp_config["N"]), desc="Splits"):
            train_ids, test_ids = load_split(exp_folder, split)
            x_train, y_train = x.loc[train_ids].copy(), y.loc[train_ids].copy()
            with open(f'{exp_folder}/split_{split}/scaler_y.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
            y_train = pd.DataFrame(loaded_scaler.transform(y_train), columns=y_train.columns)
            path_save_modules = f'{exp_folder}/split_{split}/modules'
            pbar = tqdm(exp_config["MODEL_NAMES"], desc="Models", leave=False)
            for mod_name in pbar:
                pbar.set_description(f"Running {mod_name} modules...")
                if mod_name not in ['dnn', 'cnn', 'bilstm']:
                    continue
                mod_ = modules_class_instance[mod_name]
                train_modules(mod_, hs[mod_name], x_train, y_train, path_save_modules, repetitions=100)

    # Train models
    if "train_models" in steps or 'all' in steps:
        print(f"Training models...")
        for split in tqdm(range(exp_config["N"]), desc="Splits"):
            train_ids, test_ids = load_split(exp_folder, split)
            x_train, y_train = x.loc[train_ids].copy(), y.loc[train_ids].copy()
            with open(f'{exp_folder}/split_{split}/scaler_y.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
            y_train = pd.DataFrame(loaded_scaler.transform(y_train), columns=y_train.columns)
            path_save_model = f'{exp_folder}/split_{split}/models'
            path_save_module = f'{exp_folder}/split_{split}/modules'
            pbar = tqdm(exp_config["MODEL_NAMES"], desc="Models", leave=False)
            for mod_name in pbar:
                pbar.set_description(f"Running {mod_name} model...")
                modules_paths = [f'{path_save_module}/{mod_name}_{pig}.h5' for pig in y_train.columns]
                mod_ = class_instance[mod_name]
                if mod_name in ['rf', 'xgb']:
                    hyperp = hs[mod_name]
                else:
                    hyperp = hs['concatenatedModel']
                    for h in hyperp:
                        h.update({"modules_path": modules_paths})
                train_model(mod_, hyperp, x_train, y_train, repetitions=100, path_save_model=path_save_model)

    # Compute metrics
    if "compute_metrics" in steps or 'all' in steps:
        print(f"Computing metrics...")
        for split in tqdm(range(exp_config["N"]), desc="Splits"):
            train_ids, test_ids = load_split(exp_folder, split)
            x_train, y_train = x.loc[train_ids].copy(), y.loc[train_ids].copy()
            x_test, y_test = x.loc[test_ids].copy(), y.loc[test_ids].copy()
            with open(f'{exp_folder}/split_{split}/scaler_y.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
            path_model = f'{exp_folder}/split_{split}/models'
            path_metrics = Path(f'{exp_folder}/split_{split}/metrics')
            path_metrics.mkdir(exist_ok=True, parents=True)
            for mod_name in exp_config["MODEL_NAMES"]:
                mod_ = class_instance[mod_name]
                mod_.load(path_model)
                py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(x_test)), columns=y_test.columns)
                df = mets.compute_metrics_df(y_test, py)
                df.to_csv(path_metrics / f'{mod_.name}_test.csv')
                py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(x_train)), columns=y_test.columns)
                df = mets.compute_metrics_df(y_train, py)
                df.to_csv(path_metrics / f'{mod_.name}_train.csv')

