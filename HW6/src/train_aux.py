import torch
import pytorch_lightning as pl


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg):
        super(LightningWrapper, self).__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch, batch_indx):
        pass
