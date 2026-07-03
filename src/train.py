import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_config
from src.dataset import MVTecDataset
from src.patchcore import PatchCore

# Flag an image as defective once its score exceeds the 99th percentile of the
# (defect-free) training scores. Calibrating from training data avoids the
# hand-tuned magic numbers the UI and API used to carry.
THRESHOLD_PERCENTILE = 99


def train(
    data_root: str,
    category: str,
    output_dir: str = "outputs/memory_bank",
    image_size: int = None,
    coreset_ratio: float = None,
    batch_size: int = 8,
    device: str = "cpu",
) -> PatchCore:
    config = get_config(category)
    image_size = image_size or config.image_size
    coreset_ratio = coreset_ratio or config.coreset_ratio

    print(f"\nTraining PatchCore on '{category}' (image_size={image_size}, "
          f"coreset_ratio={coreset_ratio})")

    dataset = MVTecDataset(data_root, category, train=True, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    model = PatchCore(device=device, coreset_ratio=coreset_ratio)
    model.eval()

    patches = []
    for images, _, _ in tqdm(loader, desc="Extracting features"):
        features = model.extract_features(images.to(device))
        patches.append(model._reshape_to_patches(features)[0])

    model.build_memory_bank(patches)
    print(f"Memory bank: {model.memory_bank.shape[0]} patches")

    train_scores = [model.score(img.unsqueeze(0))[0] for img, _, _ in dataset]
    model.threshold = float(np.percentile(train_scores, THRESHOLD_PERCENTILE))
    print(f"Threshold ({THRESHOLD_PERCENTILE}th pct of train scores): "
          f"{model.threshold:.3f}")

    save_dir = Path(output_dir) / category
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.memory_bank, save_dir / "memory_bank.pt")
    metadata = {
        "image_size": image_size,
        "coreset_ratio": coreset_ratio,
        "threshold": model.threshold,
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved memory bank and metadata to {save_dir}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a PatchCore memory bank for one category")
    parser.add_argument("--data_root", required=True, help="Path to MVTec dataset root")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--output_dir", default="outputs/memory_bank")
    parser.add_argument("--image_size", type=int, default=None, help="Overrides the per-category default")
    parser.add_argument("--coreset_ratio", type=float, default=None, help="Overrides the per-category default")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        args.data_root,
        args.category,
        output_dir=args.output_dir,
        image_size=args.image_size,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
