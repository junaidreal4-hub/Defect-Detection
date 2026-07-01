import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

from src.inference import load_model, predict

CATEGORIES = ["bottle", "cable", "capsule", "hazelnut", "metal_nut",
              "pill", "screw", "toothbrush", "transistor", "zipper"]

loaded_models = {}


def run_inference(image: np.ndarray, category: str):
    if category not in loaded_models:
        try:
            loaded_models[category] = load_model(category)
        except FileNotFoundError:
            return None, f"No trained model for '{category}'. Run train.py first."

    model = loaded_models[category]

    # Windows fix: write, close, then process, then delete
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()  # close immediately so Windows releases the lock

    Image.fromarray(image).save(tmp_path)
    result = predict(model, tmp_path)
    os.unlink(tmp_path)  # now safe to delete

    overlay_rgb = cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB)
    label = "DEFECTIVE ⚠️" if result["score"] > 4.3 else "NORMAL ✅"
    info  = f"**{label}**\nAnomaly score: `{result['score']}`"

    return overlay_rgb, info


with gr.Blocks(title="Visual Defect Detection") as demo:
    gr.Markdown("## 🔍 Visual Defect Detection — PatchCore on MVTec AD")
    gr.Markdown("Upload a product image and select its category. The model highlights where defects are.")

    with gr.Row():
        image_input = gr.Image(label="Input Image", type="numpy")
        heatmap_output = gr.Image(label="Anomaly Heatmap")

    category_select = gr.Dropdown(choices=CATEGORIES, value="bottle", label="Product Category")
    result_text = gr.Markdown()
    run_btn = gr.Button("Detect Defects", variant="primary")

    run_btn.click(
        fn=run_inference,
        inputs=[image_input, category_select],
        outputs=[heatmap_output, result_text],
    )

if __name__ == "__main__":
    demo.launch()