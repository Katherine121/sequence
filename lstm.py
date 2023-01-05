import torch
from einops import repeat
from torch import nn


class ResNet(nn.Module):

    def __init__(self, backbone):
        super(ResNet, self).__init__()
        self.backbone = backbone
        self.in_features = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


class LSTM(nn.Module):
    def __init__(self, *, backbone, num_classes1, num_classes2,
                 hidden_size, num_layers, len, dropout=0.):
        super().__init__()
        self.data_model = ResNet(backbone=backbone)
        self.feature_dim = self.data_model.in_features
        self.len = len
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2

        self.ang_linear = nn.Linear(num_classes2, self.feature_dim)

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bias=True, batch_first=True, dropout=dropout, bidirectional=False)

        self.mlp_head_target = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes1)
        )
        self.mlp_head_angle = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes2)
        )

    def forward(self, img, ang):
        # b,len,3,224,224->b*len,3,224,224->b*len,512->b,len,512
        img = self.data_model(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.feature_dim)

        # b,len,181->b,len,512
        ang[:, -1, :] = 0
        ang = self.ang_linear(ang)
        # b,len,512->b,len,512
        img += ang

        img, _ = self.lstm(img)
        img = img[:, -1, :]

        return self.mlp_head_target(img), self.mlp_head_angle(img)
