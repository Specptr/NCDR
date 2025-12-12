# app.py
"""
Application entrypoint
- parses CLI args (model path, optional device)
- loads the model with infer.load_model
- wraps model with a small callable and starts the Qt application

Usage:
    python -m src.app --model-path checkpoints/mlp_mnist.pth --device cuda
"""

import argparse
import sys
import os
import numpy as np
import torch

from PyQt5.QtWidgets import QApplication

from src.inference.inference_utils import load_model, predict_from_qimage

# import UI
from src.ui.ui import MainWindow

def parse_args():
    parser = argparse.ArgumentParser("MNIST Draw & Predict UI")
    parser.add_argument("--model-path", type=str, default=os.path.join("checkpoints", "mlp_mnist.pth"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--canvas-size", type=int, default=280)
    return parser.parse_args()

def main():
    args = parse_args()

    # load model
    device = args.device
    model = load_model(args.model_path, device=device, hidden_sizes=args.hidden_sizes, dropout=args.dropout)
    # prepare callable for UI: it should accept a QImage and return (probs, pred)
    def on_predict(qimage):
        # ensure we run inference on the requested device
        probs, pred = predict_from_qimage(model, qimage, device=device)
        # safety: ensure numpy array
        probs = np.asarray(probs, dtype=float)
        return probs, pred

    app = QApplication(sys.argv)
    win = MainWindow(on_predict=on_predict)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
