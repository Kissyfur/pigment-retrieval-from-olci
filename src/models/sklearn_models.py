import xgboost as xgb
import pickle

from src.models import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.data.data_utils import augment_data


class RandomForestModel(BaseModel):
    def __init__(self, name='rf'):
        super().__init__(name=name)

    def model_factory(self, **kwargs):
        model = RandomForestRegressor(**kwargs)
        return model

    def save_model(self, p):
        p = p.with_suffix('.pkl')
        with open(p, "wb") as f:
            pickle.dump(self.model, f)

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y)

    def load_model(self, p):
        p = p.with_suffix('.pkl')
        with open(p, "rb") as f:
            self.model = pickle.load(f)

    def hyperparameter_search(self, hyperparams_space, x, y, n_iter=50, random_state=42,
                              inner_splits=3, repetitions=100, **kwargs):
        model = self.model_factory(random_state=random_state)
        # x, y = augment_data(x, y, replicate=repetitions)

        randomized_search = RandomizedSearchCV(
            estimator=model, param_distributions=hyperparams_space, n_iter=n_iter,
            cv=inner_splits, random_state=random_state, n_jobs=-1)
        randomized_search.fit(x, y)
        return randomized_search.best_params_,  randomized_search.best_score_


class XGBModel(RandomForestModel):
    def __init__(self, name='xgb'):
        super().__init__(name=name)

    def fit(self, x, y, eval_set=None, **kwargs):
        return self.model.fit(x, y, eval_set=eval_set)

    def model_factory(self, **kwargs):
        model = xgb.XGBRegressor(**kwargs)
        return model

    def save_model(self, p):
        p = p.with_suffix('.json')
        self.model.save_model(p)

    def load_model(self, p):
        p = p.with_suffix('.json')
        self.model = xgb.XGBRegressor()
        self.model.load_model(p)
