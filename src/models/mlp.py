# mlp.py 25 12 11
"""
Multi-Layer Perceptron
Fully Connected Network
"""

import torch
import torch.nn as nn
from .base import BaseClassifier
from typing import List

class MLPClassifier(BaseClassifier):
    def __init__(self, input_dim: int=28*28, hidden_sizes: List[int]=[512, 256],
                 num_classes: int=10, dropout: float=0.2):
        """
        input_dim: flattened image size
        hidden_sizes: list with neuron count per hidden layer
        num_classes: number of output classes
        dropout: dropout ratio between hidden layers
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h)) # stable training
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)