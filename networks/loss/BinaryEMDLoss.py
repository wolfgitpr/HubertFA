import torch


class BinaryEMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, pred, target):
        cum_dim = -1
        cum_pred = pred.cumsum(cum_dim)
        cum_target = target.cumsum(cum_dim)
        loss = (self.loss(cum_pred, cum_target) +
                self.loss(cum_pred.flip(cum_dim), cum_target.flip(cum_dim))) / 2.0
        return loss
