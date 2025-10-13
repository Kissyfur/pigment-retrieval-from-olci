import pandas as pd
from tqdm import tqdm
from src.data.data_utils import augment_data


def train_model(model, hyperparams_space, x, y, path_save_model, repetitions=100):
    hp, loss = model.hyperparameter_search(hyperparams_space, x, y, r=42, repetitions=repetitions)
    model.build_model(**hp)
    x_aug, y_aug = augment_data(x, y, replicate=repetitions)
    model.fit(x_aug, y_aug, **hp)
    model.save(path_save_model)
    # pd.DataFrame(hp).to_csv(path_save_model + f'/{model.name}_best_hyperparam.csv')
    hp.update({'val_loss': loss})
    pd.DataFrame([hp]).to_csv(path_save_model + f'/{model.name}_best_hyperparam.csv', index=False)

    return


def train_modules(mod, hyperparams_space, x, y, path_save_modules, repetitions=100):
    hps = []
    x_aug, y_aug = augment_data(x, y, replicate=repetitions)
    for col_name in tqdm(y.columns):
        module = mod.__class__(f"{mod.name}_{col_name}")
        hp = module.hyperparameter_search(hyperparams_space=hyperparams_space, x=x, y=y[[col_name]], inner_splits=1,
                                          r=42, repetitions=repetitions, patience=4)
        hps.append(hp)
        module.build_model(**hp)
        module.fit(x_aug, y_aug[[col_name]], **hp)
        module.save(path_save_modules)
    pd.DataFrame(hps, index=y.columns).to_csv(path_save_modules + f'/{mod.name}_best_hyperparam.csv')

