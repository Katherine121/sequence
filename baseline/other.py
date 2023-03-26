import torchvision.models
from torch import nn
from torchvision.models import ResNet18_Weights, VGG16_Weights, ShuffleNet_V2_X1_0_Weights, MobileNet_V3_Small_Weights


class resnet18(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class vgg16(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(vgg16, self).__init__()
        self.backbone = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(25088, num_classes1)
        self.head_target = nn.Linear(25088, num_classes1)
        self.head_angle = nn.Linear(25088, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class shufflenet_v2(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(shufflenet_v2, self).__init__()
        self.backbone = torchvision.models.shufflenet_v2_x1_0(weights=(ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1))
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)
        self.head_target = nn.Linear(in_features, num_classes1)
        self.head_angle = nn.Linear(in_features, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(mobilenet_v3, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.head_label = nn.Linear(576, num_classes1)
        self.head_target = nn.Linear(576, num_classes1)
        self.head_angle = nn.Linear(576, num_classes2)

    def forward(self, x):
        x = self.backbone(x)
        return self.head_label(x), self.head_target(x), self.head_angle(x)
