import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.patchcore import PatchCore


def available_categories(memory_bank_dir: str = "outputs/memory_bank") -> list:
    """Categories that have a trained memory bank on disk."""
    root = Path(memory_bank_dir)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "memory_bank.pt").exists())


def load_model(category: str, memory_bank_dir: str = "outputs/memory_bank", device: str = "cpu") -> PatchCore:
    category_dir = Path(memory_bank_dir) / category
    bank_path = category_dir / "memory_bank.pt"
    if not bank_path.exists():
        raise FileNotFoundError(f"No memory bank at {bank_path}. Run train.py first.")

    metadata = json.loads((category_dir / "metadata.json").read_text())

    model = PatchCore(device=device)
    model.eval()
    model.memory_bank = torch.load(bank_path, map_location=device)
    model.threshold = metadata["threshold"]
    model.image_size = metadata["image_size"]
    return model


def predict(model: PatchCore, image_path: str, threshold: float = None):
    threshold = threshold if threshold is not None else model.threshold

    original = Image.open(image_path).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

    tensor = get_transforms(model.image_size)(original).unsqueeze(0)
    score, heatmap = model.score(tensor)

    heatmap = cv2.resize(heatmap, (original.width, original.height))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    return {
        "score": round(score, 4),
        "is_defective": score > threshold,
        "heatmap": heatmap,
        "overlay_bgr": overlay,
    }
