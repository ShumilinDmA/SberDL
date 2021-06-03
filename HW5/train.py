# Other libs
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import random
import numpy as np
from tqdm.notebook import tqdm

# MLflow
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Hydra
import hydra
from hydra import utils

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom files
from dataset.datasets import CustomDataset
from model.classification_network import TabNetClassifier
from model.sparce_loss import SparceLoss


DATA_PATH = os.path.join(os.path.abspath("."), "data/encoded_dataset.csv")


def load_data(data_path: str, seed: int, test_size: float = 0.2):
    df = pd.read_csv(data_path)
    data_train, data_valid = train_test_split(df, test_size=test_size, stratify=df['salary'], shuffle=True,
                                              random_state=seed)

    categorical_columns = ['workclass', "education", "marital-status", "occupation", 'relationship',
                           'race', 'sex', 'hours_per_week_bins', 'native_country']
    numerical_columns = ['age_log', 'fnlwgt_log', "education-num", 'capital-gain', 'capital_loss_log']
    target = 'salary'

    train_dataset = CustomDataset(data_train, numerical_columns=numerical_columns,
                                  categorical_columns=categorical_columns, target=target)
    valid_dataset = CustomDataset(data_valid, numerical_columns=numerical_columns,
                                  categorical_columns=categorical_columns, target=target)

    return train_dataset, valid_dataset


def train_model(model,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                classification_critetion,
                sparce_ctiterion,
                optimizer,
                epoches: int,
                batch_size: int,
                num_workers: int):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    for epoch in tqdm(range(epoches)):
        print(epoch)
    # TODO write loop for training and validation

    return


@hydra.main(config_name="config.yaml", config_path='./configs')
def main(cfg: DictConfig) -> None:
    # Fix all seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_dataset, valid_dataset = load_data(data_path=DATA_PATH, seed=cfg.seed, test_size=cfg.test_size)

    # MLflow block
    original_pwd = utils.get_original_cwd()
    mlflow.set_tracking_uri('file://' + original_pwd + '/mlruns')
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment(cfg.experiment_name)
    except MlflowException:  # If such experiment already exist
        experiment_id = client.get_experiment_by_name(cfg.experiment_name).experiment_id

    # Define model
    model = TabNetClassifier(n_output_classes=cfg.n_output_classes,
                             n_classification_layer=cfg.n_classification_layer,
                             num_unique_values_dict=train_dataset.num_unique_values_dict,
                             embedding_size=cfg.embedding_size,
                             n_numerical_columns=len(train_dataset.numerical_columns),
                             n_shared_layers=cfg.n_shared_layers,
                             n_decision_blocks=cfg.n_decision_blocks,
                             hidden_size=cfg.hidden_size,
                             meaningful_part=cfg.meaningful_part,
                             n_decision_steps=cfg.n_decision_steps,
                             virtual_batch_size=cfg.virtual_batch_size,
                             momentum=cfg.momentum,
                             gamma=cfg.gamma)

    # Choosing correct loss function for classification
    if cfg.n_output_classes == 1:
        classification_criterion = nn.BCEWithLogitsLoss()
    else:
        classification_criterion = nn.CrossEntropyLoss()
    sparce_criterion = SparceLoss()

    # Define specific optimizer
    optim = cfg.optimizer
    if optim.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                     betas=(optim.betas.one, optim.betas.two), weight_decay=optim.weight_decay)
    elif optim.name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                      betas=(optim.betas.one, optim.betas.two), weight_decay=optim.weight_decay)
    elif optim.name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate, weight_decay=optim.weight_decay,
                                        momentum=optim.momentum, alpha=optim.alpha)
    elif optim.name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, weight_decay=optim.weight_decay,
                                    momentum=optim.momentum, nesterov=optim.nesterov)
    else:
        raise ValueError("No such optimizer. Available optimizers are: Adam, AdamW, RMSprop, SGD")

    # TODO Add scheduler and config to it

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.run_name) as run:

        train_model(model, train_dataset=train_dataset, valid_dataset=valid_dataset,
                    classification_critetion=classification_criterion, sparce_ctiterion=sparce_criterion,
                    optimizer=optimizer, epoches=cfg.epoches, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        # TODO log all data from config and config file also


if __name__ == "__main__":
    main()
