# Other libs
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import random
import numpy as np

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

# Custom files
from dataset.datasets import CustomDataset
from model.classification_network import TabNetClassifier
from model.sparce_loss import SparceLoss
from train_aux import train_model


DATA_PATH = os.path.join(os.path.abspath("."), "data/encoded_dataset.csv")


def load_data(data_path: str, seed: int, test_size: float = 0.2):
    df = pd.read_csv(data_path)
    data_train, data_valid = train_test_split(df, test_size=test_size, stratify=df['salary'], shuffle=True,
                                              random_state=seed)
    data_train.reset_index(drop=True, inplace=True)
    data_valid.reset_index(drop=True, inplace=True)

    categorical_columns = ['workclass', "education", "marital-status", "occupation", 'relationship',
                           'race', 'sex', 'hours_per_week_bins', 'native_country']
    numerical_columns = ['age_log', 'fnlwgt_log', "education-num", 'capital-gain', 'capital_loss_log']
    target = 'salary'

    train_dataset = CustomDataset(data_train, numerical_columns=numerical_columns,
                                  categorical_columns=categorical_columns, target=target)
    valid_dataset = CustomDataset(data_valid, numerical_columns=numerical_columns,
                                  categorical_columns=categorical_columns, target=target)

    return train_dataset, valid_dataset


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
        raise NotImplementedError("No such optimizer. Available optimizers are: Adam, AdamW, RMSprop, SGD")

    schedule = cfg.scheduler
    if schedule.enable:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=schedule.factor,
                                                               patience=schedule.patience, threshold=0.0001,
                                                               cooldown=schedule.cooldown, min_lr=0, eps=1e-08,
                                                               verbose=schedule.verbose)
    else:
        scheduler = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.run_name):
        mlflow.log_param('batch_size', cfg.batch_size)
        mlflow.log_param('epoches', cfg.epoches)
        mlflow.log_param("init_learning_rate", cfg.learning_rate)
        mlflow.log_param("lambda_sparce", cfg.lambda_sparce)
        mlflow.log_param("n_decision_steps", cfg.n_decision_steps)
        mlflow.log_param("n_shared_layers", cfg.n_shared_layers)
        mlflow.log_param("n_decision_blocks", cfg.n_decision_blocks)
        mlflow.log_param("n_output_classes", cfg.n_output_classes)
        mlflow.log_param("n_classification_layer", cfg.n_classification_layer)
        mlflow.log_param("embedding_size", cfg.embedding_size)
        mlflow.log_param("hidden_size", cfg.hidden_size)
        mlflow.log_param("meaningful_part", cfg.meaningful_part)
        mlflow.log_param("virtual_batch_size", cfg.virtual_batch_size)
        mlflow.log_param("momentum", cfg.momentum)
        mlflow.log_param("gamma", cfg.gamma)
        mlflow.log_param("optimizer", optim)
        mlflow.log_param("scheduler", schedule)
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_artifact(f"{os.getcwd()}/.hydra/config.yaml")

        train_model(cfg, model, train_dataset=train_dataset, valid_dataset=valid_dataset,
                    classification_criterion=classification_criterion, sparce_criterion=sparce_criterion,
                    optimizer=optimizer, scheduler=scheduler)


if __name__ == "__main__":
    main()
