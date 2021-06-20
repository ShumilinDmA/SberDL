import os
import argparse
from omegaconf import OmegaConf

from src.data import collate_fn, load_dataset
from src.train_aux import LightningWrapper

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

import mlflow
from mlflow.tracking import MlflowClient


TEST_DATA = os.path.join(os.path.abspath("."), "data/test_encoded.pickle")


def main():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument("-r", "--run_id", required=True, type=str)
    args = parser.parse_args()
    run_id = args.run_id

    client = MlflowClient()
    run = client.get_run(run_id)

    model_uri = "runs:/{}/model".format(run.info.run_id)
    config_url = "{}/config.yaml".format(run.info.artifact_uri)
    cfg = OmegaConf.load(config_url[7:])

    seed_everything(seed=cfg.seed, workers=False)

    test_dataset = load_dataset(TEST_DATA, cfg)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             collate_fn=collate_fn)

    loaded_model = mlflow.pytorch.load_model(model_uri)
    model_wrapper = LightningWrapper(loaded_model, cfg=cfg)
    trainer = pl.Trainer(logger=False)
    trainer.test(model_wrapper, test_loader, ckpt_path="best", verbose=True)


if __name__ == "__main__":
    main()
