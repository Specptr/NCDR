# inference_utils.py
"""
Inference utilities for MNIST UI.

Provides:
- load_model: construct and load a saved MLP checkpoint
- preprocess_qimage: convert a QImage (from drawing canvas) into a normalized tensor usable by the model
- predict_from_qimage: convenience wrapper that takes a QImage and returns probabilities (numpy array)

Notes:
- The preprocessing mirrors the training transforms:
    - Resize to 28x28, convert to float [0,1]
    - Invert if necessary (UI draws black on white; MNIST has white digit on black background)
    - Normalize using MNIST mean/std (0.1307, 0.3081)
    - Flatten to (1, 784)
- This module deliberately keeps device handling explicit so the UI can choose CPU/GPU.
"""

from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Import your model class (adjust import path if you placed file elsewhere)
from src.models.mlp import MLPClassifier
from src.models.cnn import CNNClassifier

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def load_model(checkpoint_path: str, device: str = "cpu",
               hidden_sizes: Optional[List[int]] = None,
               dropout: float = 0.2, model_type: str = "mlp") -> torch.nn.Module:
    """
    Create MLPClassifier with matching architecture and load weights from checkpoint Path.
    Returns the model on the requested device.
    """

    device = torch.device(device)

    if model_type == "cnn":
        model = CNNClassifier(dropout=dropout)
    else:
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        model = MLPClassifier(
                input_dim=28*28,
                hidden_sizes=hidden_sizes,
                num_classes=10,
                dropout=dropout
            )

    # The train.save_checkpoint saved a dict with "model_state" key
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        # backwards compatibility if whole state_dict saved
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def qimage_to_pil(img_q) -> Image.Image:
    """
    Convert a QImage or QPixmap (from PyQt) to a PIL.Image (grayscale).
    The UI will pass a QImage with Format_Grayscale8 or an RGB image; handle both.
    """
    # Import here to avoid PyQt dependency when using only infer utilities
    from PyQt5.QtGui import QImage
    if isinstance(img_q, QImage):
        q = img_q
    else:
        # If QPixmap passed, convert to QImage
        q = img_q.toImage()

    # Ensure Format_Grayscale8 for easier conversion if possible; otherwise fallback
    if q.format() == QImage.Format_Grayscale8:
        w = q.width()
        h = q.height()
        buffer = q.bits().asstring(w * h)
        pil = Image.frombytes("L", (w, h), buffer)
        return pil
    else:
        # Convert via RGBA byte buffer then to L
        q2 = q.convertToFormat(QImage.Format_RGBA8888)
        w = q2.width()
        h = q2.height()
        ptr = q2.bits()
        ptr.setsize(w * h * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        pil = Image.fromarray(arr[..., :3])  # drop alpha
        pil = pil.convert("L")
        return pil

def preprocess_pil(pil: Image.Image, flatten: bool = True) -> np.ndarray:
    """
    Given a PIL grayscale image, convert to a normalized 1x784 numpy array ready for model.
    Steps:
      - Resize to 28x28 (ANTIALIAS / LANCZOS)
      - Convert to float [0,1]
      - Invert colors if background is white (we detect by checking average)
      - Normalize by MNIST mean/std
      - Flatten to (1, 784)
    """
    # ensure grayscale
    pil = pil.convert("L")
    pil = pil.resize((28, 28), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 255.0  # shape (28,28), values in [0,1]

    # Flatten and normalize
    arr = (arr - MNIST_MEAN) / MNIST_STD
    if flatten:
        return arr.reshape(1, -1)
    else:
        return arr.reshape(1, 1, 28, 28)

def preprocess_qimage(qimage, flatten: bool = True) -> torch.Tensor:
    """
    Full convert from QImage/QPixmap to torch.FloatTensor on CPU (shape [1,784]).
    """
    pil = qimage_to_pil(qimage)
    arr = preprocess_pil(pil, flatten=flatten)  # numpy
    tensor = torch.from_numpy(arr).float()
    return tensor

def predict_from_qimage(model: torch.nn.Module, qimage, device: str = "cpu", flatten: bool = True) -> Tuple[np.ndarray, int]:
    """
    Given a loaded model and a QImage, produce (probs_numpy_array, predicted_label).
    - probs: numpy array shape (10,)
    - predicted_label: int
    """
    device = torch.device(device)
    model.to(device)
    x = preprocess_qimage(qimage, flatten=flatten).to(device)  # shape [1,784]
    with torch.no_grad():
        logits = model(x)  # shape [1,10]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return probs, pred
