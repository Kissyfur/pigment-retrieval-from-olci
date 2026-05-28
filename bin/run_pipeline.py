import argparse
import json
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from pathlib import Path
from src.data.data_spliting import load_data_split, get_scaler
from src.models.model_training import train_modules, train_model
from src.metrics.metrics import Metrics
from src.models.keras_models import ConcatenatedModulesModel, ConvolutionalModel, DenseModel, BilstmModel
from src.models.sklearn_models import RandomForestModel, XGBModel


def class_instance_factory(model_name):
    if model_name == 'rf':
        return RandomForestModel()
    elif model_name == 'xgb':
        return XGBModel()
    elif model_name == 'cnn':
        return ConvolutionalModel()
    elif model_name == 'dnn':
        return DenseModel()
    elif model_name == 'bilstm':
        return BilstmModel()
    elif model_name in ['concatenatedCNN', "concatenatedDNN", "concatenatedBiLSTM"]:
        return ConcatenatedModulesModel(model_name)
    else:
        print(f"Class {model_name} not implemented")
        return None


mets = Metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training and evaluation')
    parser.add_argument('--exp_config', metavar='-e', type=str, help='models for the experiments',
                        default='test_experiment')
    parser.add_argument('--steps', metavar='-s', nargs='+', help='steps to run',
                        default=['compute_metrics'])

    args = parser.parse_args()
    steps = args.steps
    with open(args.exp_config, 'r') as f:
        exp_config = json.load(f)

    exp_folder = exp_config["EXP_NAME"]
    splits_folder = exp_config["SPLITS_PATH"]
    with open(exp_config['HYPERPARAM_SPACE_FILE'], 'r') as f:
        hs = json.load(f)

    class FunctionCallbackOnSet(tf.keras.callbacks.Callback):

        def __init__(self, x, y, scaler):
            super().__init__()
            self.x = x
            self.y = y
            self.scaler = scaler
            self.values = []

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 != 0:
                return
            py = pd.DataFrame(self.scaler.inverse_transform(self.model.predict(self.x, verbose=0)),
                              columns=exp_config["OUT_VARS_SHORT"])
            m = mets.compute_metrics_df(self.y, py).mean(axis=1)['R2']
            self.values.append(m)

    patience = exp_config['PATIENCE']
    repetitions = exp_config['REPETITIONS']
    std_x = exp_config['INP_AUG_COEFF']
    std_y = exp_config['OUT_AUG_COEFF']
    
    # Read data and preprocess
    x = pd.read_csv(exp_config["INP_PATH"], usecols=exp_config["INP_VARS"])[exp_config["INP_VARS"]]
    y = pd.read_csv(exp_config["OUT_PATH"], usecols=exp_config["OUT_VARS"])[exp_config["OUT_VARS"]]

    # y = y.loc[x["med and black sea"] == 1]
    # x = x.loc[x["med and black sea"] == 1]
    if exp_config["MEDITERRANEAN_ONLY"]:
        y = y.loc[x["med"]==1]
        x = x.loc[x["med"]==1]
    print("Sample length: ", len(x))

    y = y.rename(columns=dict(zip(exp_config["OUT_VARS"], exp_config["OUT_VARS_SHORT"])))
    print(x.head())
    print(y.head())

    # Train modules for deep learning models. This takes a lot of computing time (~12 hours for experiment_1)
    if "train_modules" in steps or 'all' in steps:
        for split in tqdm(range(exp_config["N"]), desc="Training modules in splits"):

            x_train, y_train, _, _ = load_data_split(x, y, splits_folder, split, transform=True)
            path_save_modules = f'{exp_folder}/split_{split}/modules'
            pbar = tqdm(exp_config["MODEL_NAMES"], desc="Models", leave=False)
            for mod_name in pbar:
                pbar.set_description(f"Running {mod_name} modules...")
                if mod_name not in ['dnn', 'cnn', 'bilstm']:
                    continue
                mod_ = class_instance_factory(mod_name)
                train_modules(mod_, hs[mod_name], x_train, y_train, path_save_modules, repetitions=repetitions,
                              std_x=std_x, std_y=std_y, patience=patience)

    # Train models
    if "train_models" in steps or 'all' in steps:
        for split in tqdm(range(exp_config["N"]), desc="Training models in splits"):
            x_train, y_train, x_test, y_test = load_data_split(x, y, splits_folder, split, transform=True)
            path_save_model = f'{exp_folder}/split_{split}/models'
            pbar = tqdm(exp_config["MODEL_NAMES"], desc="Models", leave=False)
            for mod_name in pbar:
                cb = None
                if y_test is not None:
                    loaded_scaler = get_scaler(splits_folder, split)
                    cb = FunctionCallbackOnSet(x_test, y_test, loaded_scaler)
                pbar.set_description(f"Running {mod_name} model...")
                mod_ = class_instance_factory(mod_name)
                hyperp = hs[mod_name]
                if mod_name not in ["xgb", "rf"]:
                    for h in hyperp:
                        h.update({"output_dim": len(y.columns)})
                if "concatenated" in mod_name:
                    for h in hyperp:
                        path_save_module = f'{h["root_path"]}/split_{split}/modules'
                        modules_paths = [f'{path_save_module}/{module_name}' for module_name in h["modules_names"]]
                        h.update({"modules_path": modules_paths})
                train_model(mod_, hyperp, x_train, y_train, repetitions=repetitions, std_x=std_x, std_y=std_y,
                            path_save_model=path_save_model, cb=cb)

    # Compute metrics
    if "compute_metrics" in steps or 'all' in steps:
        print(f"Computing metrics...")
        for split in tqdm(range(exp_config["N"]), desc="Splits"):
            x_train, y_train, x_test, y_test = load_data_split(x, y, splits_folder, split, transform=False)
            loaded_scaler = get_scaler(splits_folder, split)
            path_model = f'{exp_folder}/split_{split}/models'
            path_metrics = Path(f'{exp_folder}/split_{split}/metrics')
            path_metrics.mkdir(exist_ok=True, parents=True)
            for mod_name in exp_config["MODEL_NAMES"]:
                mod_ = class_instance_factory(mod_name)
                mod_.load(path_model)
                if x_test is not None:
                    py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(x_test)), columns=y_test.columns)
                    df = mets.compute_metrics_df(y_test, py)
                    df.to_csv(path_metrics / f'{mod_.name}_test.csv')
                py = pd.DataFrame(loaded_scaler.inverse_transform(mod_.predict(x_train)), columns=y_train.columns)
                df = mets.compute_metrics_df(y_train, py)
                df.to_csv(path_metrics / f'{mod_.name}_train.csv')
