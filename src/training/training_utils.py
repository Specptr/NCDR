# training_utils.py
"""
Utility helpers
"""

import os
import random
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(preferred: str=None) -> torch.device:
    if preferred:
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, path: str, extra: Dict[str, Any]=None):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_checkpoint(path: str, model: torch.nn.Module=None, optimizer: torch.optim.Optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return 100.0*correct / target.size(0)

def save_json(obj: Dict[str, Any], path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)