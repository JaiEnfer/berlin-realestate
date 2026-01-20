import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskResNet18(nn.Module):
    """
    Shared ResNet18 backbone.
    Head 1: room classification (5 classes)
    Head 2: quality binary classification (1 logit)
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the final fc layer, keep feature extractor
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: [B, 512, 1, 1]
        feat_dim = backbone.fc.in_features  # 512

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
        )

        self.room_head = nn.Linear(feat_dim, num_classes)
        self.quality_head = nn.Linear(feat_dim, 1)  # binary logit

    def forward(self, x):
        f = self.features(x)
        f = self.shared(f)
        room_logits = self.room_head(f)
        quality_logit = self.quality_head(f).squeeze(1)  # [B]
        return room_logits, quality_logit
