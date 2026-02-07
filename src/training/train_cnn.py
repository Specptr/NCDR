# train_cnn.py
# 25 12 11
"""
Training script for MNIST CNN
- parases args
- sets seed & device
- builds data loaders
- instantiates model, optimizer, loss
- trains
- validates
- saves best checkpoint

Usage examples:
    python -m src.training.train_cnn --epochs 10 --batch-size 256 --lr 1e-3 --device cuda
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import time
from pathlib import Path

from config import config
from .training_utils import set_seed, get_device, save_checkpoint
from src.data.data_utils import get_mnist_loaders
from src.models.cnn import CNNClassifier

def parse_args():
    p = argparse.ArgumentParser("Train a CNN on MNIST (PyTorch)")
    p.add_argument("--epochs", type=int, default=config.epochs)
    p.add_argument("--batch-size", type=int, default=config.batch_size)
    p.add_argument("--lr", type=float, default=config.lr)
    p.add_argument("--weight-decay", type=float, default=config.weight_decay)
    p.add_argument("--model-dir", type=str, default=config.model_dir)
    p.add_argument("--model-file", type=str, default="cnn_mnist.pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=config.num_workers)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=config.seed)
    return p.parse_args()

def train_one_epoch(model, device, loader, criterion, optimizer, epoch, log_interval):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=running_loss / (batch_idx + 1),
                             acc=100.0 * correct / total)

    avg_loss = running_loss / len(loader)
    avg_acc = 100.0 * correct / max(total, 1)
    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, device, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)
        val_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    val_loss /= len(loader)
    val_acc = 100.0 * correct / max(total, 1)
    return val_loss, val_acc

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # CNN: flatten=False
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, flatten=False, num_workers=args.num_workers
    )

    model = CNNClassifier(dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.model_file)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer, epoch, config.log_interval)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            print(f"New best val acc: {val_acc:.2f}% (previous {best_val_acc:.2f}%). Saving model to {model_path}")
            save_checkpoint(model, optimizer, epoch, model_path, extra={"history": history, "model_type": "cnn"})
            best_val_acc = val_acc

    # test best
    ckpt = torch.load(model_path, map_location="cpu")
    final_model = CNNClassifier(dropout=args.dropout)
    final_model.load_state_dict(ckpt["model_state"])
    final_model = final_model.to(device)
    test_loss, test_acc = validate(final_model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
