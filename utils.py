import torch
from torch import nn


class WeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num):
        super(WeightedLoss, self).__init__()
        params = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        loss1, loss2, loss3 = x
        loss1 = loss1 / (self.params[0] ** 2) + torch.log(self.params[0])
        loss2 = loss2 / (self.params[1] ** 2) + torch.log(self.params[1])
        loss3 = loss3 / (2 * (self.params[2] ** 2)) + torch.log(self.params[2])
        loss_sum += loss1 + loss2 + loss3
        return loss_sum
