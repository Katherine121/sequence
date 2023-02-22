import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Dronet(nn.Module):
    def __init__(self, img_channels, num_classes1, num_classes2):
        """
        Define model architecture.

        ## Arguments
        `img_dim`: image dimensions.
        `img_channels`: Target image channels.
        `num_classes1, num_classes2`: Dimension of model output.
        """
        super(Dronet, self).__init__()

        # get the device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.channels = img_channels
        self.conv_modules = nn.ModuleList()
        self.beta = torch.Tensor([0]).float().to(self.device)

        # Initialize number of samples for hard-mining

        self.conv_modules.append(nn.Conv2d(self.channels, 32, (5, 5), stride=(2, 2), padding=(2, 2)))
        filter_amt = np.array([32, 64, 128])
        for f in filter_amt:
            x1 = int(f / 2) if f != 32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3, 3), stride=(2, 2), padding=(1, 1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3, 3), padding=(1, 1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1, 1), stride=(2, 2)))
        # create convolutional modules
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2))

        bn_amt = np.array([32, 32, 32, 64, 64, 128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())
        self.dropout1 = nn.Dropout()

        self.cur_linear = nn.Linear(6272, num_classes1)
        self.next_linear = nn.Linear(6272, num_classes1)
        self.ang_linear = nn.Linear(6272, num_classes2)
        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()
        self.decay = 0.1

    def init_weights(self):
        '''
        intializes weights according to He initialization.
        ## parameters
        None
        '''
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        '''
        forward pass of dronet

        ## parameters
        `x`: `Tensor`: The provided input tensor`
        '''
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)

        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx + 1](x2)
            x2 = self.relu_modules[relu_idx + 1](x2)
            x2 = self.conv_modules[conv_idx + 1](x2)
            x1 = self.conv_modules[conv_idx + 2](x1)
            x3 = torch.add(x1, x2)
            x1 = x3
            bn_idx += 2
            relu_idx += 2
            conv_idx += 3

        x4 = torch.flatten(x3).reshape(-1, 6272)
        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        cur_label = self.cur_linear(x5)
        cur_label = self.sigmoid1(cur_label)

        next_label = self.next_linear(x5)
        next_label = self.sigmoid1(next_label)

        steer = self.ang_linear(x5)

        return cur_label, next_label, steer

    def loss(self, k, steer_true, steer_pred, coll_true, coll_pred):
        '''
        loss function for dronet. Is a weighted sum of hard mined mean square
        error and hard mined binary cross entropy.
        ## parameters
        `k`: `int`: the value for hard mining; the `k` highest losses will be learned first,
        and the others ignored.
        `steer_true`: `Tensor`: the torch tensor for the true steering angles. Is of shape
        `(N,1)`, where `N` is the amount of samples in the batch.
        `steer_pred`: `Tensor`: the torch tensor for the predicted steering angles. Also is of shape
        `(N,1)`.
        `coll_true`: `Tensor`: the torch tensor for the true probabilities of collision. Is of
        shape `(N,1)`
        `coll_pred`: `Tensor`: the torch tensor for the predicted probabilities of collision.
        Is of shape `(N,1)`
        '''
        # for steering angle
        mse_loss = self.hard_mining_mse(k, steer_true, steer_pred)
        # for collision probability
        bce_loss = self.beta * (self.hard_mining_entropy(k, coll_true, coll_pred))
        return mse_loss + bce_loss

    def hard_mining_mse(self, k, y_true, y_pred):
        '''
        Compute Mean Square Error for steering
        evaluation and hard-mining for the current batch.
        ### parameters

        `k`: `int`: number of samples for hard-mining
        `y_true`: `Tensor`: torch Tensor of the expected steering angles.

        `y_pred`: `Tensor`: torch Tensor of the predicted steering angles.
        '''
        loss_steer = (y_true - y_pred) ** 2

        # hard mining
        # get value of k that is minimum of batch size or the selected value of k
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_steer, k=k_min, dim=0)
        max_loss_steer = torch.gather(loss_steer, dim=0, index=indices)
        # mean square error
        hard_loss_steer = torch.div(torch.sum(max_loss_steer), k_min)
        return hard_loss_steer

    def hard_mining_entropy(self, k, y_true, y_pred):
        '''
        computes binary cross entropy for probability collisions and hard-mining.
        ## parameters
        `k`: `int`: number of samples for hard-mining
        `y_true`: `Tensor`: torch Tensor of the expected probabilities of collision.
        `y_pred`: `Tensor`: torch Tensor of the predicted probabilities of collision.
        '''

        loss_coll = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_coll, k=k_min, dim=0)
        max_loss_coll = torch.gather(loss_coll, dim=0, index=indices)
        hard_loss_coll = torch.div(torch.sum(max_loss_coll), k_min)
        return hard_loss_coll