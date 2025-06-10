import pickle
import numpy as np
import itertools
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def to_dataframe_lat_lon(x_ar):
    return  x_ar.to_dataframe().reset_index()


def transform_x(x_train, x_test, transformation):
    if "scaled" in transformation:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    if "pca" in transformation:
        pca = PCA()
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test


def stratified_kfold_multidim_kmeans(x, y, n_splits=10, clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    clusters = kmeans.fit_predict(y)

    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return split.split(x, clusters)


def stratified_split_multidim_kmeans(x, y, clusters=10, test_size=0.2, random_state=42):
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    clusters = kmeans.fit_predict(y)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return split.split(x, clusters)


def stratified_split_multidim_proj_1d(x, y, proj_1d, q=10, test_size=0.2, random_state=42):
    y_proj = proj_1d(y)
    bins = pd.qcut(y_proj, q=q, labels=False)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return split.split(x, bins)


def fill_value_2d(arr, val, percent):
    arr_c = arr.copy()
    shape = arr.shape
    num_changes = int(np.prod(shape) * percent)
    all_coords = list(itertools.product(*[range(s) for s in shape]))
    random_coords = np.array(all_coords)[np.random.choice(len(all_coords), num_changes, replace=False)]
    arr_c[random_coords[:, 0], random_coords[:, 1]] = val
    return arr_c


def fill_random_2d(arr, percent):
    arr_c = arr.copy()
    shape = arr.shape
    num_changes = int(np.prod(shape) * percent)
    all_coords = list(itertools.product(*[range(s) for s in shape]))
    random_coords = np.array(all_coords)[np.random.choice(len(all_coords), num_changes, replace=False)]
    arr_c[random_coords[:, 0], random_coords[:, 1]] = np.random.randint(-1, 50, len(random_coords))
    return arr_c


# def augment_data(x_train, y_train, replicate=10):
#     x_aug = pd.DataFrame(x_train.values.repeat(replicate, axis=0), columns=x_train.columns)
#     y_aug = pd.DataFrame(y_train.values.repeat(replicate, axis=0), columns=y_train.columns)
#
#     n = x_aug.shape[0]
#
#     # transformations:
#     std_dev = 2.5 / 100
#     # std_dev = 5. / 100
#     x_noise = np.random.normal(1, std_dev, x_aug.shape)
#     x_scale_factor = np.random.normal(1, std_dev, (n, 1))
#
#     y_noise = np.random.normal(1, std_dev, y_aug.shape)
#
#     mixing_factor = np.random.uniform(0.9, 1., (n, 1))
#     mixing_indices = np.random.choice(range(n), n)
#
#     # Apply transformations:
#     x_aug.loc[:] = mixing_factor * x_aug.values + (1 - mixing_factor) * x_aug.loc[mixing_indices].values
#     y_aug.loc[:] = mixing_factor * y_aug.values + (1 - mixing_factor) * y_aug.loc[mixing_indices].values
#
#     x_aug *= x_noise
#     x_aug *= x_scale_factor
#
#     y_aug *= y_noise
#     return pd.concat([x_train, x_aug], ignore_index=True), pd.concat([y_train, y_aug], ignore_index=True)


def augment_data(x_train, y_train, replicate=10, seed=42):
    np.random.seed(seed)
    # x_aug = pd.DataFrame(x_train.values.repeat(replicate, axis=0), columns=x_train.columns)
    # y_aug = pd.DataFrame(y_train.values.repeat(replicate, axis=0), columns=y_train.columns)
    x_aug = np.repeat(x_train.values, replicate, axis=0)
    y_aug = np.repeat(y_train.values, replicate, axis=0)
    n = x_aug.shape[0]

    # transformations:
    std_dev = 2.5 / 100
    # std_dev = 5. / 100
    #  = np.random.normal(1, std_dev, x_aug.shape)
    #  = np.random.normal(1, std_dev, (n, 1))

    # y_noise = np.random.normal(1, std_dev, y_aug.shape)

    mixing_factor = np.random.uniform(0.9, 1., (n, 1))
    mixing_indices = np.random.choice(n, n)

    # Apply transformations:
    x_aug = mixing_factor * x_aug + (1 - mixing_factor) * x_aug[mixing_indices]
    y_aug = mixing_factor * y_aug + (1 - mixing_factor) * y_aug[mixing_indices]

    # Add independent Gaussian noise per wavelength (feature)
    x_aug += np.random.normal(0.0, std_dev, x_aug.shape)

    # Add global Gaussian noise per sample (shifts all wavelengths equally)
    x_aug += np.random.normal(0.0, std_dev,  (n, 1))

    # x_aug *= x_scale_factor

    # y_aug *= y_noise
    y_aug += np.random.normal(0.0, std_dev, y_aug.shape)

    # return pd.concat([x_train, x_aug], ignore_index=True), pd.concat([y_train, y_aug], ignore_index=True)
    return (
        pd.DataFrame(np.vstack([x_train.values, x_aug]), columns=x_train.columns),
        pd.DataFrame(np.vstack([y_train.values, y_aug]), columns=y_train.columns)
    )

def inverse_transform(py_transformed, y_var, model_path):
    path_transf = model_path / "data_transformations"
    if 'scaled_log_pigments' == y_var:
        path_transf = path_transf / 'log_pigments_scaler.pkl'
        with open(path_transf, 'rb') as f:
            transf = pickle.load(f)
        log_py = transf.inverse_transform(py_transformed)
        return np.exp(log_py) - 0.001
    elif 'scaled_pigments' == y_var:
        path_transf = path_transf / 'pigments_scaler.pkl'
        with open(path_transf, 'rb') as f:
            transf = pickle.load(f)
        return transf.inverse_transform(py_transformed)
    elif 'log_pigments' == y_var:
        return np.exp(py_transformed) # - 0.001
    else:
        return py_transformed

