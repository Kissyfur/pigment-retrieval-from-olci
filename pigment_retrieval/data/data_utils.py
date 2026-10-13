import pickle
import numpy as np
import itertools
import pandas as pd
import xarray as xr

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def to_xarray_dims(df, dims):
    # Set the multi-index for xarray
    df = df.set_index(dims)
    # Convert to xarray
    return df.to_xarray()


def to_dataframe_lat_lon(x_ar, dims):
    x_aux = x_ar.reset_coords(dims)
    dims_aux = [dim + '_dim' for dim in dims]
    x_aux = x_aux.rename_dims(dict(zip(dims, dims_aux)))
    return x_aux.to_dataframe().reset_index()

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
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    clusters = kmeans.fit_predict(y)

    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return split.split(x, clusters)


def stratified_split_multidim_kmeans(x, y, n_splits=1, clusters=10, test_size=0.2, random_state=42):
    kmeans = KMeans(n_clusters=clusters, random_state=random_state)
    y = y.fillna(y.mean(axis=0)).copy()
    clusters = kmeans.fit_predict(y)

    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
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


def augment_data(x_train, y_train, std_x, std_y, replicate=10, seed=42):
    np.random.seed(seed)
    x_aug = np.repeat(x_train.values, replicate, axis=0)
    y_aug = np.repeat(y_train.values, replicate, axis=0)
    n = x_aug.shape[0]

    mixing_factor = np.random.uniform(0.9, 1., (n, 1))
    mixing_indices = np.random.choice(n, n)

    # Apply transformations:
    x_aug = mixing_factor * x_aug + (1 - mixing_factor) * x_aug[mixing_indices]
    y_aug = mixing_factor * y_aug + (1 - mixing_factor) * y_aug[mixing_indices]

    # Add independent Gaussian noise per feature
    x_aug += np.random.normal(0.0, std_x, x_aug.shape)
    y_aug += np.random.normal(0.0, std_y, y_aug.shape)

    return (
     pd.DataFrame(np.vstack([x_train.values, x_aug]), columns=x_train.columns),
     pd.DataFrame(np.vstack([y_train.values, y_aug]), columns=y_train.columns)
    )


def replicate_data(darr, dim='Id', replicate=10):
    return xr.concat([darr] * replicate, dim=dim)


def apply_multiplicative_noise(darr, std=0.1, std_dims=None, seed=42):
    return apply_noise(darr, std, std_dims, seed, add=False)


def apply_additive_noise(darr, std=0.1, std_dims=None, seed=42):
    return apply_noise(darr, std, std_dims, seed, add=True)


def apply_noise(darr, std=0.1, std_dims=None, seed=42, add=True):
    if std_dims is None:
        return darr
    np.random.seed(seed)
    # Add independent Gaussian noise per feature
    x_dim_sizes = tuple(darr.sizes[d] for d in std_dims)
    if add:
        noise_x = np.random.normal(0.0, std, size=x_dim_sizes)
        noise_x = xr.DataArray(noise_x, dims=std_dims)
        return darr + noise_x
    else:
        noise_x = np.random.normal(1.0, std, size=x_dim_sizes)
        noise_x = xr.DataArray(noise_x, dims=std_dims)
        return darr * noise_x


# def augment_data(x, y, repetitions, std_x, std_dims_x, std_y, std_dims_y):
    # replicate_data
#    x_aug = replicate_data(x, dim='Id', replicate=repetitions)
#    y_aug = replicate_data(y, dim='Id', replicate=repetitions)

    # apply noise
#    y_aug = apply_additive_noise(y_aug, std=std_y, std_dims=std_dims_y)
#    x_aug = apply_additive_noise(x_aug, std=std_x, std_dims=std_dims_x)

#     return xr.concat([x, x_aug], dim='Id'), xr.concat([y, y_aug], dim='Id')

def standardize_data(x_train, x_test, dims=('Id', 'lat', 'lon', 'time')):
    # standardize
    mean = x_train.mean(dim=dims)
    std  = x_train.std(dim=dims)

    x_train = (x_train - mean) / std
    x_test  = (x_test - mean) / std
    return x_train, x_test


def random_paired_combinations(darr_1, darr_2, dim='Id', factor=0.9, seed=42):
    np.random.seed(seed)
    n = len(darr_1[dim].values)

    mixing_factor = np.random.uniform(factor, 1., n)
    mixing_factor = xr.DataArray(mixing_factor, dims=dim)
    mixing_indices = np.random.choice(n, n)

    # Apply transformations:
    darr_1 = mixing_factor * darr_1 + ((1 - mixing_factor) * darr_1.isel(Id=mixing_indices)).values
    darr_2 = mixing_factor * darr_2 + ((1 - mixing_factor) * darr_2.isel(Id=mixing_indices)).values

    return darr_1, darr_2


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


