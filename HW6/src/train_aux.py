import torch
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, f1
from hydra import utils
import mlflow


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg=None):
        super(LightningWrapper, self).__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        pred = self.model((x_cat, x_num))
        loss = self.criterion(pred, y)
        y_softmax = torch.softmax(pred, dim=-1).detach()
        y_pred = torch.argmax(y_softmax, dim=-1)
        y_proba = y_softmax[:, 1]
        return {"loss": loss, "y_pred": y_pred, "y_true": y, "y_proba": y_proba}

    def validation_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        pred = self.model((x_cat, x_num))
        loss = self.criterion(pred, y)
        y_softmax = torch.softmax(pred, dim=-1).detach()
        y_pred = torch.argmax(y_softmax, dim=-1)
        y_proba = y_softmax[:, 1]
        return {"val_loss": loss, "y_pred": y_pred, "y_true": y, "y_proba": y_proba}

    def test_step(self, batch, batch_idx):
        x_cat, x_num, y = batch
        pred = self.model((x_cat, x_num))
        loss = self.criterion(pred, y)
        y_softmax = torch.softmax(pred, dim=-1).detach()
        y_pred = torch.argmax(y_softmax, dim=-1)
        y_proba = y_softmax[:, 1]
        return {"test_loss": loss, "y_pred": y_pred, "y_true": y, "y_proba": y_proba}

    def training_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_pred'].view(-1) for x in outputs])
        y_true = torch.cat([x['y_true'].view(-1) for x in outputs])
        y_proba = torch.cat([x['y_proba'].view(-1) for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        roc_auc = float(auroc(preds=y_proba, target=y_true, pos_label=1))
        f1_score = float(f1(preds=y_proba, target=y_true))
        acc = float(accuracy(preds=y_hat, target=y_true))

        mlflow.log_metric("train_loss", float(avg_loss), step=self.current_epoch)
        mlflow.log_metric("lr", float(self.get_lr()), step=self.current_epoch)
        mlflow.log_metric("train_acc", acc, step=self.current_epoch)
        mlflow.log_metric("train_f1_score", f1_score, step=self.current_epoch)
        mlflow.log_metric("train_roc_auc", roc_auc, step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_pred'].view(-1) for x in outputs])
        y_true = torch.cat([x['y_true'].view(-1) for x in outputs])
        y_proba = torch.cat([x['y_proba'].view(-1) for x in outputs])
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

        roc_auc = float(auroc(preds=y_proba, target=y_true, pos_label=1))
        f1_score = float(f1(preds=y_proba, target=y_true))
        acc = float(accuracy(preds=y_hat, target=y_true))
        self.log('val_roc_auc', roc_auc)

        mlflow.log_metric("val_loss", float(avg_loss), step=self.current_epoch)
        mlflow.log_metric("val_acc", acc, step=self.current_epoch)
        mlflow.log_metric("val_f1_score", f1_score, step=self.current_epoch)
        mlflow.log_metric("val_roc_auc", roc_auc, step=self.current_epoch)

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_pred'].view(-1) for x in outputs])
        y_true = torch.cat([x['y_true'].view(-1) for x in outputs])
        y_proba = torch.cat([x['y_proba'].view(-1) for x in outputs])
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        roc_auc = float(auroc(preds=y_proba, target=y_true, pos_label=1))
        f1_score = float(f1(preds=y_proba, target=y_true))
        acc = float(accuracy(preds=y_hat, target=y_true))
        avg_loss = float(avg_loss)

        self.log('test_loss', avg_loss)
        self.log("test_accuracy", acc)
        self.log("test_f1_score", f1_score)
        self.log("test_roc_auc", roc_auc)

    def configure_optimizers(self):
        self.optimizer = utils.instantiate(self.cfg.optimizer, params=self.model.parameters())

        if self.cfg.enable_scheduler:
            scheduler = utils.instantiate(self.cfg.scheduler, optimizer=self.optimizer)
            lr_schedulers = {'scheduler': scheduler, 'monitor': 'val_loss'}
            return [self.optimizer], [lr_schedulers]

        return self.optimizer

    def get_callbacks(self):
        callbacks = []

        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=".",
                                                  filename="best_model",
                                                  monitor="val_roc_auc",
                                                  verbose=True,
                                                  save_top_k=1,
                                                  save_weights_only=False,
                                                  mode='max')
        callbacks.append(checkpoint)

        if self.cfg.enable_early_stopping:
            early_stopping = utils.instantiate(self.cfg.early_stopping)
            callbacks.append(early_stopping)

        return callbacks

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
