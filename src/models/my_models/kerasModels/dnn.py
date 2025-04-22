import keras
import xarray as xr
import tensorflow as tf
import numpy as np

from pathlib import Path
from . import KerasBaseModel


class DNN(KerasBaseModel):
    class_name = 'DNN'

    def __init__(self, path=None, name=class_name, **kwargs):
        self.hl_dim = kwargs["hl_dim"]
        self.layers_num = kwargs["layers_num"]
        self.dropout = kwargs["dropout"]
        super().__init__(path=path, name=name, **kwargs)

    def build_model(self):
        model = keras.Sequential()
        for i in range(self.layers_num):
            model.add(keras.layers.Dense(self.hl_dim[i],
                                         kernel_initializer="glorot_uniform",
                                         activation='relu' if i != self.layers_num - 1 else 'linear'))
            if i != self.layers_num - 1:
                model.add(keras.layers.Dropout(self.dropout))
        self.model = model


class DNN_th(DNN):
    class_name = 'DNN_th'

    def __init__(self, path=None, name=class_name, **kwargs):
        self.threshold = kwargs["threshold"]
        super().__init__(path=path, name=name, **kwargs)

    def build_model(self):
        model = keras.Sequential()
        for i in range(self.layers_num):
            model.add(keras.layers.Dense(self.hl_dim[i],
                                         kernel_initializer="glorot_uniform",
                                         activation='relu' if i != self.layers_num - 1 else 'linear'))
            if i != self.layers_num - 1:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Lambda(lambda x: tf.maximum(x, self.threshold)))
        self.model = model


class DNN_pretrain(DNN):
    class_name = 'DNN_pretrain'

    def __init__(self, path=None, name=class_name,**kwargs):
        self.hl_dim = kwargs["hl_dim"]
        self.out_dims = kwargs["out_dims"]
        self.hidden_layers_num = kwargs["hidden_layers_num"]
        self.hidden_layers_names = kwargs["hidden_layers_names"]
        self.mini_wavelength_indices = [1, 2, 3, 5, 8]
        self.olci_wavelength_indices = [0, 4, 6, 7, 9, 10, 11, 12]
        self.dnn_mini_name = 'submodel_5_wv'
        self.dnn_mini = DNN(name=self.dnn_mini_name, pca_components=5, do_log_x=True,
                            do_log_y=True, polynomial_degree=1, wavelengths=[412, 442, 490, 560, 673])
        super().__init__(path=path, name=name, **kwargs)

    def transfer_weights(self):
        self.model.get_layer("embed_mini").set_weights(self.dnn_mini.model.get_layer("embed").get_weights())
        self.model.get_layer("hidden").set_weights(self.dnn_mini.model.get_layer("hidden").get_weights())
        self.model.get_layer("output").set_weights(self.dnn_mini.model.get_layer("output").get_weights())

    def build_model(self):
        if self.pca_components is not None:
            inp_dim = self.pca_components + self.dnn_mini.pca_components
        else:
            inp_dim = len(self.x_wavelengths)
            inp_dim += len(self.variables) if self.variables else 0

        inp_layer = keras.Input(shape=inp_dim)

        mini_indices = list(range(self.dnn_mini.pca_components))
        olci_indices = list(range(self.dnn_mini.pca_components, inp_dim))
        selected_mini = tf.gather(inp_layer, mini_indices, axis=1)
        selected_olci = tf.gather(inp_layer, olci_indices, axis=1)

        dense1_mini = keras.layers.Dense(self.hl_dim, name='embed_mini', kernel_initializer='normal',
                                         activation='linear')(selected_mini)
        dense1_olci = keras.layers.Dense(self.hl_dim, name='embed_olci', kernel_initializer='normal',
                                         activation='linear')(selected_olci)

        add_layer = keras.layers.Add()([dense1_mini, dense1_olci])
        add_relu_layer = keras.layers.ReLU()(add_layer)

        dense2 = keras.layers.Dense(self.hl_dim, name='hidden', kernel_initializer='normal',
                                    activation='relu')(add_relu_layer)

        output = keras.layers.Dense(self.out_dims, name='output', kernel_initializer='normal',
                                    activation='linear',
                                    # activity_regularizer=keras.regularizers.L2(0.001)
                                    )(dense2)

        model = keras.Model(inp_layer, output)
        self.model = model
        self.model.build(inp_dim)

    def predict(self, x):
        x_ext = self.extract_x_variables(x)
        ampl_indices_olci = self.olci_wavelength_indices + [13, 14]
        x_olci = x_ext[:, ampl_indices_olci]
        x_mini = x_ext[:, self.mini_wavelength_indices]
        x_olci = self.x_transform.fit_transform(x_olci) if self.polynomial_trans else x_olci
        x_olci = self.pca.transform(x_olci) if self.do_pca else x_olci
        x_mini = self.dnn_mini.pca.transform(x_mini)
        x_ext = np.hstack([x_mini, x_olci])
        py = self.model.predict(x_ext)
        return np.exp(py) if self.do_log_y else py

    def fit(self, x, y, batch, epochs, lr, val, **kwargs):
        # Pretraining
        print(f"Pretraining...")
        x_supp, y_supp = xr.load_dataset(kwargs["x_data_2"]), xr.load_dataset(kwargs["y_data_2"])
        self.dnn_mini.fit(x_supp, y_supp, epochs=1000, batch=512, lr=lr, val=val)
        self.transfer_weights()

        # Prepare data
        x_train = self.extract_x_variables(x)
        ampl_indices_olci = self.olci_wavelength_indices + [13, 14]
        x_train_olci = x_train[:, ampl_indices_olci]
        x_train_mini = x_train[:, self.mini_wavelength_indices]
        x_train_olci = self.x_transform.fit_transform(x_train_olci) if self.polynomial_trans else x_train_olci
        self.fit_pca(x_train_olci)
        x_train_olci = self.pca.transform(x_train_olci) if self.do_pca else x_train_olci
        x_train_mini = self.dnn_mini.pca.transform(x_train_mini)
        x_train = np.hstack([x_train_mini, x_train_olci])
        y_train = self.extract_y_variables(y)
        y_train = np.log(y_train + self.eps) if self.do_log_y else y_train

        # Remaining parameters training
        self.freeze_mini_model()
        print(f"Warm up...")
        # hist = super().fit_model(x_train, y_train, batch, epochs, lr, val, kwargs)
        hist = {'pretraining': super().fit_model(x_train, y_train, batch, epochs, lr, val, kwargs)}

        # Fine-tuning
        self.unfreeze_mini_model()
        print(f"Fine-tuning...")
        hist.update(super().fit_model(x_train, y_train, batch, epochs, lr=1e-5, val=val, kwargs=kwargs))
        hist.update(self.compute_metrics(x, y))

        return hist
    def freeze_mini_model(self):
        self.model.get_layer("embed_mini").trainable = False
        self.model.get_layer("hidden").trainable = False
        self.model.get_layer("output").trainable = False

    def unfreeze_mini_model(self):
        self.model.get_layer("embed_mini").trainable = True
        self.model.get_layer("hidden").trainable = True
        self.model.get_layer("output").trainable = True

    def save(self, path):
        super().save(path)
        p = Path(path) / self.name
        self.dnn_mini.save(p)

    def load(self, path):
        super().load(path)
        p = Path(path) / self.name
        self.dnn_mini.load(p)
