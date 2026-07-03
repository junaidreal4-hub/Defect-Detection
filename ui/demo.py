import os
import tempfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.inference import available_categories, load_model, predict

CATEGORIES = available_categories()
_models = {}


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

    run_btn.click(
        fn=run_inference,
        inputs=[image_input, category_select],
        outputs=[heatmap_output, result_text],
    )

if __name__ == "__main__":
    demo.launch()
