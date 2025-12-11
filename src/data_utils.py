# data_utils.py
"""
Data utilities
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def flatten_tensor(t):
        return t.view(-1)

def get_default_transforms(flatten: bool=True):
    """
    Return a composed torchvision transform for MNIST
    If flatten is True, the transform will flatten the 28x28 image to a 1D tensor
    suitable for an MLP
    For CNNs, if flatten is False, it returns a CxHxW tensor
    """
    basic = [
        transforms.ToTensor(), # (C=1, H=28, W=28) -> [0,1]
        transforms.Normalize((0.1307,), (0.3081,)) # standard MNIST mean & std
    ]
    if flatten:
        basic.append(transforms.Lambda(flatten_tensor))
    return transforms.Compose(basic)

def get_mnist_loaders(batch_size: int=128, flatten: bool=True, num_workers: int=4):
    transform = get_default_transforms(flatten=flatten)

    # Download train dataset and test
    train_full = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Create a validation split from trian_full
    val_size = 5000
    train_size = len(train_full) - val_size
    train_set, val_set = torch.utils.data.random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader