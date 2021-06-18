import pickle
import numpy as np
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything

from src.utils.data import CustomDataset, collate_fn

# MLflow
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Hydra
import hydra
from hydra import utils

TRAIN_DATA = "data/train_encoded.pickle"
VALID_DATA = "data/valid_encoded.pickle"
TEST_DATA = "data/test_encoded.pickle"

with open(TRAIN_DATA, "rb") as file:
    train_df = pickle.load(file)
    train_df.sort_values(by='Date', inplace=True)

with open(VALID_DATA, "rb") as file:
    valid_df = pickle.load(file)
    valid_df.sort_values(by='Date', inplace=True)


cat_columns = ["Card", "Use Chip", "MCC", "Errors", "Outcome", "is_diff_merchant",
               "is_diff_merchant_city", "is_diff_merchant_state"]
target_columns = ["IsFraud_target"]
drop_columns = ["User", "Date"]
num_columns = np.setdiff1d(train_df.columns.tolist(), cat_columns+target_columns+drop_columns).tolist()

assert len(train_df.columns) == len(cat_columns+target_columns+drop_columns+num_columns)


@hydra.main(config_name="config.yaml", config_path='./configs')
def main(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=False)

    num_uniq_embeddings = 138  # Constant
    n_numerical_col = len(num_columns)  # Constant
    feature_dim = n_numerical_col + len(cat_columns) * cfg.embedding_dim

    original_pwd = utils.get_original_cwd()
    mlflow.set_tracking_uri('file://' + original_pwd + '/mlruns')
    client = MlflowClient()

    try:
        experiment_id = client.create_experiment(cfg.experiment_name)
    except MlflowException:  # If such experiment already exist
        experiment_id = client.get_experiment_by_name(cfg.experiment_name).experiment_id

    train_dataset = CustomDataset(train_df, cat_columns=cat_columns,
                                  num_columns=num_columns,
                                  target_columns=target_columns,
                                  drop_columns=drop_columns,
                                  max_latest_seq=cfg.max_seq_len_latest)

    valid_dataset = CustomDataset(valid_df, cat_columns=cat_columns,
                                  num_columns=num_columns,
                                  target_columns=target_columns,
                                  drop_columns=drop_columns,
                                  max_latest_seq=cfg.max_seq_len_latest)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn)

    model = utils.instantiate(cfg.models, num_uniq_embeddings=num_uniq_embeddings,
                              n_numerical_col=n_numerical_col, feature_dim=feature_dim)

    optimizer = utils.instantiate(cfg.optimizer, params=model.parameters())

    if cfg.scheduler_enable:
        scheduler = utils.instantiate(cfg.scheduler, optimizer=optimizer)

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.run_name):
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("max_seq_len", cfg.max_seq_len_latest)
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_param("learning_rate", cfg.learning_rate)
        mlflow.log_param("embedding_dim", cfg.embedding_dim)
        mlflow.log_param("scheduler_enable", cfg.scheduler_enable)
        mlflow.log_param("optimizer", cfg.optimizer._target_)
        mlflow.log_params({"optimizer_params": cfg.optimizer})


if __name__ == "__main__":
    main()
