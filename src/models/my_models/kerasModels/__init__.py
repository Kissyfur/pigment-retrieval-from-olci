import keras

from src.models.my_models import Model
from src.models.my_models.kerasModels.losses import Loss, coeff_determination_loss


class KerasBaseModel(Model):
    class_name = 'KerasBaseModel'

    def __init__(self, path=None, name=class_name, **kwargs):
        self.loss = Loss(kwargs["loss"]).loss_function
        self.lr = kwargs['lr']
        self.patience = kwargs['patience'] if 'patience' in kwargs else None
        self.epochs = kwargs['epochs']
        self.batch = kwargs['batch']
        super().__init__(path=path, name=name, **kwargs)

    def fit_model(self, x, y, **kwargs):
        history = keras.callbacks.History()
        cb = [keras.callbacks.EarlyStopping(patience=self.patience, monitor="loss", restore_best_weights=True), history]
        self.model.compile(loss=self.loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr))
        return self.model.fit(x=x, y=y, batch_size=self.batch, epochs=self.epochs, callbacks=cb, verbose=0).history

    def predict(self, x):
        py = self.model.predict(x, verbose=0)
        return py

    def save_model(self, path):
        self.model.save_weights(path / f'{self.name}.weights.h5', overwrite=True)

    def load_model(self, path):
        self.model.load_weights(path / f'{self.name}.weights.h5')

