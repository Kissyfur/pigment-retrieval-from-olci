import logging

from pathlib import Path

logging.basicConfig(level=logging.INFO)


class Model:

    def __init__(self, name='baseModel', **kwargs):

        self.name = name
        self.model = None

        self.info = kwargs
        self.build_model()

    def build_model(self):
        return

    def predict(self, x):
        py = self.model.predict(x)
        return py

    def fit(self, x, y, **kwargs):
        return self.fit_model(x, y, **kwargs)

    def fit_model(self, x, y, batch, epochs, lr, val, **kwargs):
        self.model.fit(x, y)
        return {}

    def save_model(self, p):
        return

    def save(self, path):
        p = Path(path) / self.name
        p.mkdir(parents=True, exist_ok=True)
        self.save_model(p)

    def load_model(self, p):
        return

    def load(self, path):
        p = Path(path) / self.name
        self.load_model(p)