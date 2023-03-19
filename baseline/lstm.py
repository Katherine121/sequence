import torch
from einops import repeat
from torch import nn


class Extractor(nn.Module):
    def __init__(self, backbone):
        super(Extractor, self).__init__()
        self.backbone = backbone
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x


class LSTM(nn.Module):
    def __init__(self, *, backbone, extractor_dim,
                 num_classes1, num_classes2,
                 dim, num_layers, len):
        super().__init__()
        self.extractor = Extractor(backbone)
        self.extractor_dim = extractor_dim
        self.dim = dim
        self.len = len

        self.ang_linear = nn.Linear(2, dim)
        # extractor_dim+dim
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
            # cbapd
            nn.LayerNorm(dim),
            nn.Hardtanh(),
            nn.Linear(dim, dim // 2),
            nn.Hardtanh(),
            nn.Linear(dim // 2, num_classes2)
        )

    def forward(self, img, ang):
        # 试试使用HOG特征
        # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
        img = self.extractor(img.view(-1, 3, 224, 224))
        img = img.view(-1, self.len, self.extractor_dim)

        # b,len,2
        for i in range(1, self.len):
            ang[:, i, :] += ang[:, i - 1, :]
        ang = self.ang_linear(ang)

        # b,len,extractor_dim+dim
        img = torch.cat((img, ang), dim=-1)
        # b,len,extractor_dim+dim->b,len,dim
        img = self.img_linear(img)

        img, _ = self.lstm(img)
        img = img[:, -1, :]

        img = self.mlp_head(img)
        ang = img[:, self.dim:]
        img = img[:, 0: self.dim]

        return self.head_label(img), self.head_target(img), self.head_angle(ang)
