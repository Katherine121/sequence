import numpy as np
import timm.models
import torch
import torchvision.models
from torch import nn
from torchvision.models import Swin_T_Weights, \
    ResNet18_Weights, EfficientNet_B3_Weights, \
    ShuffleNet_V2_X1_0_Weights, MobileNet_V3_Small_Weights


class swint(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        Swin Transformer tiny.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(swint, self).__init__()
        self.backbone = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        """
        forward pass of swint.
        :param x: the provided input tensor.
        :return: the current position, the next position, the direction angle.
        """
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class vit(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        Vision Transformer b-16.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(vit, self).__init__()
        self.backbone = timm.models.vit_small_patch16_224(pretrained=True)
        in_features = 768
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of vit.
        :param x: the provided input tensor.
        :return: the current position.
        """
        x = self.backbone(x)
        return self.head_label(x)


class resnet18(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        ResNet18.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of resnet18.
        :param x: the provided input tensor.
        :return: the current position.
        """
        x = self.backbone(x)
        return self.head_label(x)


class efficientb3(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        EfficientNetb3.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(efficientb3, self).__init__()
        self.backbone = torchvision.models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = 1536
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        """
        forward pass of efficientb3.
        :param x: the provided input tensor.
        :return: the current position, the next position, the direction angle.
        """
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class Dronet(nn.Module):
    def __init__(self, img_channels, num_classes1, num_classes2):
        """
        Dronet.
        :param img_channels: image channels.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
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

        # initialize number of samples for hard-mining
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

        self.head_label = nn.Linear(6272, num_classes1)
        self.head_target = nn.Linear(6272, num_classes1)
        self.head_angle = nn.Linear(6272, num_classes2)
        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()
        self.decay = 0.1

    def init_weights(self):
        """
        intializes weights according to He initialization.
        :return:
        """
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        """
        forward pass of Dronet.
        :param x: the provided input tensor.
        :return: the current position, the next position, the direction angle.
        """
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

        id = self.head_label(x5)
        id = self.sigmoid1(id)

        target = self.head_target(x5)
        target = self.sigmoid1(target)

        ang = self.head_angle(x5)

        return id, target, ang


class shufflenet_v2(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        ShuffleNetV2.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(shufflenet_v2, self).__init__()
        self.backbone = torchvision.models.shufflenet_v2_x1_0(weights=(ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1))
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        """
        forward pass of shufflenet_v2.
        :param x: the provided input tensor.
        :return: the current position, the next position, the direction angle.
        """
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        MobileNetV3.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(mobilenet_v3, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = 576
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        """
        forward pass of mobilenet_v3.
        :param x: the provided input tensor.
        :return: the current position, the next position, the direction angle.
        """
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class Extractor(nn.Module):
    def __init__(self, backbone):
        """
        Feature Extractor.
        :param backbone: backbone of Feature Extractor (MobileNetV3 small).
        """
        super(Extractor, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        """
        forward pass of Extractor.
        :param x: the provided input tensor.
        :return: the visual semantic features of an image.
        """
        x = self.backbone(x)
        return x


class LSTM(nn.Module):
    def __init__(self, *, backbone, extractor_dim,
                 num_classes1, num_classes2,
                 dim, num_layers, len):
        """
        baseline (MobileNetV3 small + LSTM).
        :param backbone: backbone of Feature Extractor (MobileNetV3 small).
        :param extractor_dim: output dimension of Feature Extractor.
        :param num_classes1: output dimension of baseline.
        :param num_classes2: output dimension of baseline.
        :param dim: input dimension of LSTM.
        :param num_layers: depth of LSTM.
        :param len: input sequence length of baseline.
        """
        super().__init__()
        self.extractor = Extractor(backbone)
        self.extractor_dim = extractor_dim
        self.dim = dim
        self.len = len

        self.ang_linear = nn.Linear(2, dim)
        self.img_linear = nn.Linear(self.extractor_dim + dim, dim)

        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers, batch_first=True)

        self.mlp_head = nn.Linear(dim, 2 * dim)
        self.head_label = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_target = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes1)
        )
        self.head_angle = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Hardtanh(),
            nn.Linear(dim, dim // 2),
            nn.Hardtanh(),
            nn.Linear(dim // 2, num_classes2)
        )

    def forward(self, img, ang):
        """
        forward pass of baseline.
        :param img: input frame sequence.
        :param ang: input angle sequence.
        :return: the current position, the next position, the direction angle.
        """
        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # b,len,2->b,len,dim
        for i in range(1, self.len):
            ang[:, i, :] += ang[:, i - 1, :]
        ang = self.ang_linear(ang)

        # b,len,extractor_dim+dim->b,len,dim
        img = torch.cat((img, ang), dim=-1)
        img = self.img_linear(img)

        # b,len,dim->b,dim
        img, _ = self.lstm(img)
        img = img[:, -1, :]

        # b,dim->b,2*dim->2,b,dim
        img = self.mlp_head(img)
        ang = img[:, self.dim:]
        img = img[:, 0: self.dim]

        return self.head_label(img), self.head_target(img), self.head_angle(ang)
