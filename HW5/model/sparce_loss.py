import torch
import torch.nn as nn


class SparceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(SparceLoss, self).__init__()
        self.eps = eps

    def forward(self, masks):
        postprocessed = -masks*torch.log(masks + self.eps) / (masks.shape[0] * masks.shape[1])
        loss = postprocessed.sum()
        return loss
