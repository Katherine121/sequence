import torchvision.models
from torch import nn


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(mobilenet_v3, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(576, num_classes1)
        self.head_target = nn.Linear(576, num_classes1)
        self.head_angle = nn.Linear(576, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class resnet(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(resnet, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(self.in_features, num_classes1)
        self.head_target = nn.Linear(self.in_features, num_classes1)
        self.head_angle = nn.Linear(self.in_features, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)
