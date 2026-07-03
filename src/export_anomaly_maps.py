"""Export per-pixel anomaly maps as float32 TIFFs for the MVTec evaluation script.

Output layout mirrors the dataset:
  outputs/anomaly_maps/<category>/test/<defect_type>/<image_id>.tiff

Usage:
  python -m src.export_anomaly_maps --data_root data --category bottle
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from src.dataset import get_transforms
from src.inference import load_model


def export_maps(data_root: str, category: str, output_dir: str = "outputs/anomaly_maps", device: str = "cpu"):
    print(f"\nExporting anomaly maps for '{category}'")

    model = load_model(category, device=device)
    transform = get_transforms(model.image_size)

    test_dir = Path(data_root) / category / "test"
    save_base = Path(output_dir) / category / "test"

    for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        save_dir = save_base / defect_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(sorted(defect_dir.glob("*.png")), desc=f"  {defect_dir.name}"):
            image = Image.open(img_path).convert("RGB")
            _, heatmap = model.score(transform(image).unsqueeze(0))

            # Upsample to the original resolution: the evaluation compares against
            # full-resolution ground-truth masks.
            heatmap = np.array(
                Image.fromarray(heatmap.astype(np.float32)).resize(image.size, Image.BILINEAR),
                dtype=np.float32,
            )
            tifffile.imwrite(str(save_dir / (img_path.stem + ".tiff")), heatmap)

    print(f"Done. Maps saved to {save_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export anomaly heatmaps as TIFF files")
    parser.add_argument("--data_root", required=True, help="Path to MVTec dataset root")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--output_dir", default="outputs/anomaly_maps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    export_maps(args.data_root, args.category, args.output_dir, device=args.device)
