# models/logo_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models

class LogoEncoder(nn.Module):
    def __init__(self, backbone="resnet18", output_dim=512):
        super().__init__()

        if backbone == "resnet18":
            model = models.resnet18(pretrained=True)
            modules = list(model.children())[:-1]  # remove FC
            self.backbone = nn.Sequential(*modules)
            self.output_dim = model.fc.in_features

        elif backbone == "mobilenet_v3":
            model = models.mobilenet_v3_small(pretrained=True)
            self.backbone = model.features
            self.output_dim = model.classifier[0].in_features

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x
