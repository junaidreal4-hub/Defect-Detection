import io
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from src.inference import available_categories, load_model, predict

app = FastAPI(title="Defect Detection API", version="1.0")

_model_cache = {}


def get_model(category: str):
    if category not in available_categories():
        raise HTTPException(status_code=404, detail=f"No trained model for category: {category}")
    if category not in _model_cache:
        _model_cache[category] = load_model(category)
    return _model_cache[category]


def run_predict(model, upload: UploadFile):
    image = Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    # predict() reads from a path; stage the upload in a temp file we control
    # rather than trusting the client-supplied filename.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        return predict(model, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/categories")
def list_categories():
    return {"categories": available_categories()}


@app.post("/predict/{category}")
async def predict_defect(category: str, file: UploadFile = File(...)):
    result = run_predict(get_model(category), file)
    return JSONResponse({
        "category": category,
        "anomaly_score": result["score"],
        "is_defective": bool(result["is_defective"]),
    })


@app.post("/predict/{category}/heatmap")
async def predict_heatmap(category: str, file: UploadFile = File(...)):
    """Return the heatmap overlay as a JPEG."""
    result = run_predict(get_model(category), file)
    _, buffer = cv2.imencode(".jpg", result["overlay_bgr"])
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
