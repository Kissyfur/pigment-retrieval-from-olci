import pandas as pd
import pickle

from pathlib import Path
from src.data.data_utils import stratified_split_multidim_kmeans


def load_split(path_splits, split):
    p = Path(path_splits) / f'split_{split}' / 'data'

    with open(p / 'test_ids', "rb") as f:
         test = pickle.load(f)
    with open(p / 'train_ids', "rb") as f:
         train = pickle.load(f)
    return train, test


def get_scaler(splits_folder, split):
    with open(f'{splits_folder}/split_{split}/scaler_y.pkl', 'rb') as f:
        return pickle.load(f)


def load_data_split(x, y, splits_folder, split, transform=True):
    train_ids, test_ids = load_split(splits_folder, split)
    x_train, y_train = x.iloc[train_ids].copy(), y.iloc[train_ids].copy()
    loaded_scaler = get_scaler(splits_folder, split)
    if transform:
        y_train = pd.DataFrame(loaded_scaler.transform(y_train), columns=y_train.columns)
    x_test, y_test = None, None
    if test_ids is not None:
        x_test, y_test = x.iloc[test_ids].copy(), y.iloc[test_ids].copy()
    # print(y_train)
    # print(y_test)
    return x_train, y_train, x_test, y_test


def create_splits(x, y, n_splits=6, clusters=15, test_size=0.15, r=42, path_save_splits=None):
    # legacy version
    root_p = Path(path_save_splits)
    root_p.mkdir(parents=True, exist_ok=True)

    if test_size == 0:
        split = [(range(len(x)), None)]
    else:
        split = stratified_split_multidim_kmeans(x, y, n_splits=n_splits, clusters=clusters, test_size=test_size,
                                                 random_state=r)

    for i, (id_train, id_test) in enumerate(split):
        # x_train, y_train = x.iloc[id_train, :].copy(), y.iloc[id_train, :].copy()
        # x_test, y_test = x.iloc[id_test, :].copy(), y.iloc[id_test, :].copy()
        # print(id_test)
        p = root_p / f'split_{i}/data'
        p.mkdir(parents=True, exist_ok=True)
        print(f"Saving split in {p}...")
        with open(p / 'train_ids', "wb") as f:
            pickle.dump(id_train, f)
        with open(p / 'test_ids', "wb") as f:
            pickle.dump(id_test, f)