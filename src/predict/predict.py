import json
import numpy as np
import pickle
import pandas as pd
import xarray as xr

from pathlib import Path

MODEL_NAMES = ['concatenatedCNN', 'cnn', 'concatenatedDNN', 'dnn', 'concatenatedBiLSTM', 'bilstm', 'rf', 'xgb']


def available_models(configs_dir='experiments_config'):
    """Trained models that can predict, as {'<experiment>/<model>': (experiment config, splits with the model)}."""
    models = {}
    for path_config in sorted(Path(configs_dir).glob('*.json'), key=str):
        with open(path_config, 'r') as f:
            exp_config = json.load(f)
        for model_name in MODEL_NAMES:
            splits = [split for split in range(exp_config['N'])
                      if any(Path(exp_config['EXP_NAME'], f'split_{split}', 'models').glob(f'{model_name}.*'))
                      and Path(exp_config['SPLITS_PATH'], f'split_{split}', 'scaler_y.pkl').exists()]
            if splits:
                models[f'{Path(exp_config["EXP_NAME"]).name}/{model_name}'] = exp_config, splits
    return models


class Predictor:
    """Pigment concentrations [mg m^-3] from Rrs [sr^-1], as the mean of the model trained in each split."""

    def __init__(self, model='OLCI_sat_ft/concatenatedCNN'):
        # Imported here so that listing the available models does not load TensorFlow
        from src.models.model_training import class_instance_factory

        models = available_models()
        if model not in models:
            raise ValueError(f'{model} is not available. The available models are: {", ".join(models)}')
        exp_config, splits = models[model]
        self.bands, self.pigments = exp_config['INP_VARS'], exp_config['OUT_VARS_SHORT']
        self.models, self.scalers = [], []
        for split in splits:
            mod_ = class_instance_factory(model.split('/')[1])
            mod_.load(Path(exp_config['EXP_NAME'], f'split_{split}', 'models'))
            self.models.append(mod_)
            with open(Path(exp_config['SPLITS_PATH'], f'split_{split}', 'scaler_y.pkl'), 'rb') as f:
                self.scalers.append(pickle.load(f))

    def predict(self, data, logged=False, floor_quantile=None):
        """Predict the pigments of a DataFrame with one column per band, or of an xarray Dataset with one variable
        per band (e.g. an image with lat and lon dimensions), as a DataFrame or a Dataset with the same samples.

        The models work with ln(Rrs): give it directly with `logged`. Samples with missing or non-positive Rrs are
        not predicted, unless `floor_quantile` is given: then, as in the paper, the Rrs below that quantile of the
        positive values of their band are raised to it.
        """
        if isinstance(data, xr.Dataset):
            # Flatten the pixels of all the samples, predict them at once and give them back their dimensions
            rrs = data[self.bands].to_array('band').transpose(..., 'band')
            grid = rrs.isel(band=0, drop=True)
            x = pd.DataFrame(rrs.values.reshape(-1, len(self.bands)), columns=self.bands)
            py = self.predict(x, logged, floor_quantile)
            out = grid.to_dataset(name='rrs').drop_vars('rrs')
            for pigment in py:
                out[pigment] = (grid.dims, py[pigment].to_numpy().reshape(grid.shape), {'units': 'mg m-3'})
            return out

        x = data[self.bands].astype(float)
        if not logged:
            if floor_quantile is not None:
                x = x.clip(lower=x.where(x > 0).quantile(floor_quantile), axis=1)
            x = np.log(x.where(x > 0))
        valid = np.isfinite(x).all(axis=1)

        py = pd.DataFrame(np.nan, index=x.index, columns=self.pigments)
        if valid.any():
            log_py = [scaler.inverse_transform(mod_.predict(x[valid]))
                      for mod_, scaler in zip(self.models, self.scalers)]
            py.loc[valid] = np.exp(np.mean(log_py, axis=0))
        return py
