import os
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from src.data import get_loader
from src.train_aux import LightningWrapper

# MLflow
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Hydra
import hydra
from hydra import utils

TRAIN_DATA = os.path.join(os.path.abspath("."), "data/train_encoded.pickle")
VALID_DATA = os.path.join(os.path.abspath("."), "data/valid_encoded.pickle")
NUM_UNIQUE_EMBEDDINGS = 138
N_NUMERICAL_COLUMNS = 12
N_CATEGORICAL_COLUMNS = 8


@hydra.main(config_name="config.yaml", config_path='./configs')
def main(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=False)

    feature_dim = N_NUMERICAL_COLUMNS + N_CATEGORICAL_COLUMNS * cfg.embedding_dim

    original_pwd = utils.get_original_cwd()
    mlflow.set_tracking_uri('file://' + original_pwd + '/mlruns')
    # mlflow.set_tracking_uri(f'sqlite:////{original_pwd}/mlruns.db')  # To run mlflow logging in sqlite database

    client = MlflowClient()

    try:
        # To log artifact to local machine in artifact_location
        # experiment_id = client.create_experiment(cfg.experiment_name, artifact_location='file://'
        #                                                                                 + original_pwd + '/mlruns')
        experiment_id = client.create_experiment(cfg.experiment_name)
    except MlflowException:  # If such experiment already exist
        experiment_id = client.get_experiment_by_name(cfg.experiment_name).experiment_id

    train_loader = get_loader(TRAIN_DATA, cfg, shuffle=True)
    valid_loader = get_loader(VALID_DATA, cfg, shuffle=False)

    model = utils.instantiate(cfg.models, num_uniq_embeddings=NUM_UNIQUE_EMBEDDINGS,
                              n_numerical_col=N_NUMERICAL_COLUMNS, feature_dim=feature_dim)

    model_wrapper = LightningWrapper(model=model, cfg=cfg)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         callbacks=model_wrapper.get_callbacks(),
                         num_sanity_val_steps=0)

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.run_name):
        # Artifact and model
        mlflow.log_artifact(f"{os.getcwd()}/.hydra/config.yaml")
        mlflow.log_params(cfg.models)

        # Training parameters
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("max_seq_len", cfg.max_seq_len_latest)
        mlflow.log_param("max_epochs", cfg.max_epochs)
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_param("learning_rate", cfg.learning_rate)
        mlflow.log_param("embedding_dim", cfg.embedding_dim)

        # Optimizer
        mlflow.log_param("optimizer", cfg.optimizer._target_)
        mlflow.log_params({"optimizer_params": cfg.optimizer})

        # Scheduler
        mlflow.log_param("enable_scheduler", cfg.enable_scheduler)
        if cfg.enable_scheduler:
            mlflow.log_params({"scheduler_param": cfg.scheduler})

        # Early stopping callback
        mlflow.log_param("enable_early_stopping", cfg.enable_early_stopping)
        if cfg.enable_early_stopping:
            mlflow.log_params({"early_stopping_params": cfg.early_stopping})

        trainer.fit(model_wrapper, train_loader, valid_loader)
        trainer.validate(val_dataloaders=valid_loader, ckpt_path='best')

        # Load best model and log it
        model_wrapper.load_from_checkpoint("best_model.ckpt", model=model)
        mlflow.pytorch.log_model(model_wrapper.model, "model", requirements_file=f'{original_pwd}/requirements.txt')


if __name__ == "__main__":
    main()
