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


def create_splits(x, y, n_splits=6, clusters=15, test_size=0.15, r=42, path_save_splits=None):
    # legacy version
    root_p = Path(path_save_splits)
    root_p.mkdir(parents=True, exist_ok=True)

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