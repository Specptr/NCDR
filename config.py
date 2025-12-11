# config.py
"""
Global configuration
"""

from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "checkpoints"
    model_filename: str = "mlp_mnist.pth"
    seed: int = 42
    weight_decay: float = 1e-5
    num_workers: int = 4
    log_interval: int = 100

config = TrainConfig()