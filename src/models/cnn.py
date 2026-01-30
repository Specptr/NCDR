# cnn.py
# 26 1 30
import torch
import torch.nn as nn
from .base import BaseClassifier

class CNNClassifier(BaseClassifier):
    """
    Simple CNN for MNIST: input [B,1,28,28] -> logits [B,10]
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 64*7*7
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
