# base.py
"""
Abstract base class for classifiers
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseClassifier(torch.nn.Module, ABC):
    """
    Base Classifier extends torch.nn.Module
    and defines helper methods that
    concrete models should implemet / inherit
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError # Must implement in subclasses

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilities using softmax from logits returned by forward()
        Move model to eval() and returns CPU tensor for easy downstream use
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
        if was_training:
            self.train()
        return probs

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str=None):
        loc = map_location or "cpu"
        self.load_state_dict(torch.load(path, map_location=loc))