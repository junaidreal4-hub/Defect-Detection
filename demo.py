import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from huggingface_hub import snapshot_download

# Fetch the trained memory banks at startup. If the download fails (network,
# hub outage), still launch the UI: available_categories() will simply come up
# empty and the app reports "no trained model" instead of crashing.
try:
    snapshot_download(
        repo_id="Junaidreal4/defect-detection-weights",
        repo_type="model",
        local_dir="outputs/memory_bank",
    )
except Exception as exc:
    print(f"WARNING: could not download weights: {exc}")

from src.inference import available_categories, load_model, predict

CATEGORIES = available_categories()
_models = {}

# Sample inputs bundled with the weights repo, so a visitor can try the demo
# without sourcing their own MVTec image. Each entry is (filename, category);
# any file missing after the download is dropped rather than breaking launch.
_EXAMPLE_DIR = Path("outputs/memory_bank/examples")
EXAMPLES = [
    [str(_EXAMPLE_DIR / f), c]
    for f, c in [
        ("bottle_broken.png", "bottle"),
        ("hazelnut_hole.png", "hazelnut"),
        ("metal_nut_flip.png", "metal_nut"),
        ("wood_hole.png", "wood"),
        ("leather_glue.png", "leather"),
        ("bottle_good.png", "bottle"),
    ]
    if (_EXAMPLE_DIR / f).exists()
]


def run_inference(image: np.ndarray, category: str):
    if category not in _models:
        try:
            _models[category] = load_model(category)
        except FileNotFoundError:
            return None, f"No trained model for '{category}'. Run train.py first."

    # Stage the image on disk for predict(); close the handle before reading it
    # back so Windows doesn't hold a lock on the file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    Image.fromarray(image).save(tmp_path)
    try:
        result = predict(_models[category], tmp_path)
    finally:
        os.unlink(tmp_path)

    overlay_rgb = cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB)
    label = "DEFECTIVE ⚠️" if result["is_defective"] else "NORMAL ✅"
    info = f"**{label}**\nAnomaly score: `{result['score']}`"
    return overlay_rgb, info


with gr.Blocks(title="Visual Defect Detection") as demo:
    gr.Markdown("## 🔍 Visual Defect Detection — PatchCore on MVTec AD")
    gr.Markdown("Upload a product image and select its category. The model highlights where defects are.")

    with gr.Row():
        image_input = gr.Image(label="Input Image", type="numpy")
        heatmap_output = gr.Image(label="Anomaly Heatmap")

    category_select = gr.Dropdown(choices=CATEGORIES, value=CATEGORIES[0] if CATEGORIES else None,
                                  label="Product Category")
    result_text = gr.Markdown()
    run_btn = gr.Button("Detect Defects", variant="primary")

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[image_input, category_select],
            label="Try an example (no image handy? click one)",
        )
        gr.Markdown("_Sample images from the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) "
                    "(CC BY-NC-SA 4.0)._")

    run_btn.click(
        fn=run_inference,
        inputs=[image_input, category_select],
        outputs=[heatmap_output, result_text],
    )

if __name__ == "__main__":
    demo.launch()
