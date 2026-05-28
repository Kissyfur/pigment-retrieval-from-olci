import pandas as pd
from tqdm import tqdm
from src.data.data_utils import augment_data
from src.models.keras_models import ConcatenatedModulesModel, ConvolutionalModel, DenseModel, BilstmModel
from src.models.sklearn_models import RandomForestModel, XGBModel

def class_instance_factory(model_name):
    if model_name == 'rf':
        return RandomForestModel(),
    elif model_name == 'xgb':
        return XGBModel(),
    elif model_name == 'cnn':
        return ConvolutionalModel(),
    elif model_name == 'dnn':
        return DenseModel(),
    elif model_name == 'bilstm':
        return BilstmModel()
    elif model_name in ['concatenatedCNN', "concatenatedDNN", "concatenatedBiLSTM"]:
        return ConcatenatedModulesModel(model_name)
    else:
        print(f"Class {model_name} not implemented")
        return None


def train_model(model, hyperparams_space, x, y, path_save_model, repetitions=100, std_x=0.05, std_y=0.18, cb=None):
    x_aug, y_aug = augment_data(x, y, replicate=repetitions, std_x=std_x, std_y=std_y)
    if len(hyperparams_space) > 1:
        hp, loss = model.hyperparameter_search(hyperparams_space, x_aug, y_aug, r=42, repetitions=repetitions, patience=2)
    else:
        hp, loss = hyperparams_space[0], None
    model.build_model(**hp)
    model.fit(x_aug, y_aug, cb=cb, **hp)
    model.save(path_save_model)
    # pd.DataFrame(hp).to_csv(path_save_model + f'/{model.name}_best_hyperparam.csv')
    hp.update({'val_loss': loss})
    if cb is not None:
        hp.update({'test_error': cb.values})
    pd.DataFrame([hp]).to_csv(path_save_model + f'/{model.name}_best_hyperparam.csv', index=False)

    return


def train_modules(mod, hyperparams_space, x, y, path_save_modules, patience, repetitions=100, std_x=0.05, std_y=0.18):
    for h in hyperparams_space:
        h.update({"output_dim": 1})
    hps = []
    x_aug, y_aug = augment_data(x, y, replicate=repetitions, std_x=std_x, std_y=std_y)

    for col_name in tqdm(y.columns,  leave=False):
        module = mod.__class__(f"{mod.name}_{col_name}")
        hp, loss = module.hyperparameter_search(hyperparams_space=hyperparams_space, x=x_aug, y=y_aug[[col_name]],
                                                inner_splits=3, r=42, repetitions=repetitions, patience=patience)
        module.build_model(**hp)
        module.fit(x_aug, y_aug[[col_name]], **hp)
        module.save(path_save_modules)
        hp.update({'val_loss': loss})
        hps.append(hp)
    pd.DataFrame(hps, index=y.columns).to_csv(path_save_modules + f'/{mod.name}_best_hyperparam.csv')

