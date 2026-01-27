import json
import argparse
import pandas as pd
import pickle

from pathlib import Path
from src.data.data_spliting import create_splits, load_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training and evaluation')
    parser.add_argument('--exp_config', metavar='-e', type=str, help='models for the experiments',
                        default='test_experiment')
    args = parser.parse_args()
    with open(args.exp_config, 'r') as f:
        exp_config = json.load(f)

    exp_folder = f'{exp_config["EXP_NAME"]}'
    x = pd.read_csv(exp_config["INP_PATH"])
    y = pd.read_csv(exp_config["OUT_PATH"])

    # y = y.loc[x["med and black sea"] == 1]
    # x = x.loc[x["med and black sea"] == 1]
    if exp_config["MEDITERRANEAN_ONLY"]:
        y = y.loc[x["med"] == 1]
        x = x.loc[x["med"] == 1]
    print("Sample length: ", len(x))

    x = x[exp_config["INP_VARS"]]
    y = y[exp_config["OUT_VARS"]]
    y = y.rename(columns=dict(zip(exp_config["OUT_VARS"], exp_config["OUT_VARS_SHORT"])))

    create_splits(x=x.copy(), y=y.copy(), n_splits=exp_config["N"],
                  clusters=exp_config["CLUSTERS"], test_size=exp_config["TEST_SIZE"], r=exp_config["R"],
                  path_save_splits=exp_folder)

    for split in tqdm(range(exp_config["N"]), desc="Splits"):
        train_ids, test_ids = load_split(exp_folder, split)
        x_train, y_train = x.iloc[train_ids].copy(), y.iloc[train_ids].copy()
        scaler_y = StandardScaler()
        scaler_y.fit(y_train)
        path_save_scaler = Path(f'{exp_folder}/split_{split}/')
        path_save_scaler.mkdir(parents=True, exist_ok=True)
        # Save the scaler
        with open(path_save_scaler / 'scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)