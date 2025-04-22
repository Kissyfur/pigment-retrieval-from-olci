import keras

from keras import backend as K


def coeff_determination_loss(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)))
    return 1 - SS_res / (SS_tot + 1e-7)


def mean_absolute_percentage_exponential_error(y_true, y_pred):
    y_true = K.exp(y_true)
    y_pred = K.exp(y_pred)
    return keras.losses.mean_absolute_percentage_error(y_true, y_pred)


def mean_absolute_logarithmic_error(y_true, y_pred):
    y_true = K.log(y_true)
    y_pred = K.log(y_pred)
    return keras.losses.mean_absolute_error(y_true, y_pred)


def mean_absolute_exponential_error(y_true, y_pred):
    y_true = K.exp(y_true)
    y_pred = K.exp(y_pred)
    return keras.losses.mean_absolute_error(y_true, y_pred)

class Loss:
    def __init__(self, custom_loss):
        self.loss_dict = {
            "mae": "mae",
            "mape": "mape",
            "mse": "mse",
            "msle": "msle",
            "male": mean_absolute_logarithmic_error,
            "mapee": mean_absolute_percentage_exponential_error,
            "maee": mean_absolute_exponential_error
            # "meale": mean_exponential_absolute_logarithmic_error,
            # "r2": keras.metrics.R2Score()
        }
        self.loss_function = self.loss_dict[custom_loss]
