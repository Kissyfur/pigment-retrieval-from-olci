import keras
import numpy as np
import tensorflow as tf
import random
import copy
from sklearn.model_selection import KFold

from src.models import BaseModel

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ConvolutionalModel(BaseModel):
    def __init__(self, name='cnn'):
        self.batch = 512
        super().__init__(name=name)

    def model_factory(self, dims, kernel_sizes, output_dim, dropout=0., reg_factor=0.01, loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001), seed=42, **kwargs):
        # Define the model
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        model = keras.Sequential(name=self.name)
        model.add(keras.layers.Normalization(name='normalization'))
        model.add(keras.layers.Reshape((-1, 1)))
        for i, dim in enumerate(dims):
            model.add(keras.layers.Conv1D(filters=dim, kernel_size=kernel_sizes[i], padding='same', activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(reg_factor)))
            model.add(keras.layers.BatchNormalization())
            if dropout != 0 and i > 0:
                model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(output_dim, kernel_initializer="glorot_uniform",
                                     activation='linear', kernel_regularizer=keras.regularizers.l2(reg_factor)))

        # Compile the model
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def hyperparameter_search(self, hyperparams_space, x, y, inner_splits=3, r=42,
                              repetitions=100, patience=5, **kwargs):

        # inner_loop = stratified_split_multidim_kmeans(x, y, n_splits=inner_splits, clusters=5, test_size=0.15,
        #                                               random_state=r)

        inner_loop = KFold(n_splits=inner_splits, shuffle=True, random_state=r).split(x)
        metrics_inner_loop = []
        epochs_inner_loop = []
        for id_train, id_val in inner_loop:
            x_train, y_train = x.iloc[id_train, :].copy(), y.iloc[id_train].copy()
            # x_train, y_train = augment_data(x_train, y_train, replicate=repetitions)
            x_val, y_val = x.iloc[id_val, :].copy(), y.iloc[id_val].copy()
            hists = []
            for mod_conf in hyperparams_space:
                m_ = self.__class__()
                m_.build_model(**mod_conf)
                hists.append(m_.fit(x_train, y_train, x_val=x_val, y_val=y_val, patience=patience, **mod_conf, **kwargs))
            metrics_inner_loop.append([hist.history['val_loss'][-patience] for hist in hists])
            epochs_inner_loop.append([len(hist.history['loss']) - patience for hist in hists])
        metrics_ = np.mean(metrics_inner_loop, axis=0)
        epochs_ = np.mean(epochs_inner_loop, axis=0)
        best_indx = np.argmin(metrics_)
        best_hp = copy.deepcopy(hyperparams_space[best_indx])
        best_epochs = int(epochs_[best_indx])
        if "epochs" not in best_hp.keys():
            best_hp.update({"epochs": best_epochs})
        return best_hp, np.min(metrics_)

    def fit(self, x, y, x_val=None, y_val=None, epochs=5000, cb=None, patience=5, verbose=0, **kwargs):
        self.model.get_layer(name='normalization').adapt(x)
        if cb is None:
            cb = []
        val_data = None
        if x_val is not None:
            val_data = (x_val, y_val)
            if patience!=0:
                cb += [keras.callbacks.EarlyStopping(patience=patience)]
        h = self.model.fit(x, y, validation_data=val_data, epochs=epochs, shuffle=True, verbose=verbose,
                           batch_size=self.batch, callbacks=cb)
        return h

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)

    def save_model(self, p):
        p = p.with_suffix('.h5')
        for layer in self.model.layers:
            layer.trainable = True
        self.model.compile()
        self.model.save(p)

    def load_model(self, p):
        p = p.with_suffix('.h5')
        self.model = keras.models.load_model(p)


class BilstmModel(ConvolutionalModel):
    def __init__(self, name='bilstm'):
        super().__init__(name=name)

    def model_factory(self, dims, output_dim, dropout=0, reg_factor=0.001, loss='mse',
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001), seed=42, **kwargs):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        model = keras.Sequential(name=self.name)
        model.add(keras.layers.Normalization(name='normalization'))
        model.add(keras.layers.Reshape((-1, 1)))
        for i, dim in enumerate(dims):
            r_s = i < (len(dims) - 1)
            model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(dim, return_sequences=r_s, kernel_regularizer=keras.regularizers.l2(reg_factor),
                                  dropout=dropout)))
            model.add(keras.layers.LayerNormalization())
        model.add(keras.layers.Dense(output_dim, kernel_initializer="glorot_uniform",
                                     activation='linear', kernel_regularizer=keras.regularizers.l2(reg_factor)))
        model.compile(loss=loss, optimizer=optimizer)

        return model


class DenseModel(ConvolutionalModel):
    def __init__(self, name='dnn'):
        super().__init__(name=name)

    def model_factory(self, dims, output_dim, dropout=0, reg_factor=0.001, loss='mse',
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001), seed=42, **kwargs):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        model = keras.Sequential(name=self.name)
        model.add(keras.layers.Normalization(name='normalization'))
        for dim in dims:
            model.add(keras.layers.Dense(dim, kernel_initializer="glorot_uniform", activation='relu',
                                         kernel_regularizer=keras.regularizers.l2(reg_factor)))
            model.add(keras.layers.Dropout(dropout))
            # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(output_dim, kernel_initializer="glorot_uniform", activation='linear',
                                     kernel_regularizer=keras.regularizers.l2(reg_factor)))
        model.compile(loss=loss, optimizer=optimizer)
        return model


class ConcatenatedModulesModel(ConvolutionalModel):
    def __init__(self, name='concatenatedModel'):
        super().__init__(name=name)

    def model_factory(self, modules_path, output_dim=13, activation='linear', reg_factor=0.01,
                      last_layer=-2, loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0001), **kwargs):

        models = self.load_from_modules(modules_path=modules_path)
        # Assume all models expect the same input shape
        input_layer = keras.Input(shape=models[0].input_shape[1:])

        # Collect outputs from truncated models
        outputs = []
        for m in models:
            truncated = keras.Model(inputs=m.input, outputs=m.layers[last_layer].output)
            outputs.append(truncated(input_layer))

        # Concatenate outputs
        x = keras.layers.Concatenate()(outputs)
        # x = keras.layers.Dense(output_dim, activation='relu')(x)
        final_output = keras.layers.Dense(output_dim, activation=activation,
                                          kernel_regularizer=keras.regularizers.l2(reg_factor))(x)

        m = keras.Model(inputs=input_layer, outputs=final_output, name=self.name)
        m.compile(loss=loss, optimizer=optimizer)
        return m

    def hyperparameter_search(self, hyperparams_space, x, y, inner_splits=1, r=42,
                              repetitions=100, patience=4, **kwargs):
        # super().hyperparameter_search(hyperparams_space, x, y, inner_splits, r, repetitions, patience)
        return hyperparams_space[0], None

    def fit(self, x, y, x_val=None, y_val=None, epochs_warm_up=300, epochs=600, patience=5,
            cb=None, **kwargs):
        if cb is None:
            cb = []
        val_data = None
        if x_val is not None:
            val_data = (x_val, y_val)
            if patience!=0:
                cb += [keras.callbacks.EarlyStopping(patience=patience)]
        for layer in self.model.layers:
            layer.trainable = False
        self.model.layers[-1].trainable = True
        h = self.model.fit(x, y, validation_data=val_data, epochs=epochs, shuffle=True, batch_size=self.batch,
                           verbose=0, callbacks=cb)
        for layer in self.model.layers:
            layer.trainable = True
        # if epochs != 0:
        #     h = self.model.fit(x, y, validation_data=val_data, epochs=epochs, shuffle=True, batch_size=self.batch,
        #                        verbose=0, callbacks=cb)
        return h
    @staticmethod
    def load_from_modules(modules_path):
        models_ = []
        for mn in modules_path:
            models_.append(keras.models.load_model(mn))
        return models_
        # return self.model_factory(None, models_, output_dim, activation='linear', reg_factor=reg_factor,
        #                           last_layer=last_layer)
