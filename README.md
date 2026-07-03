# Visual Defect Detection — PatchCore on MVTec AD

Unsupervised anomaly detection for industrial inspection. A PatchCore model is
fit per product category using only defect-free images; at test time it scores
each image and produces a pixel-level heatmap localising anomalies.

The implementation uses a frozen ImageNet backbone (Wide-ResNet-50-2), a
coreset-subsampled memory bank of normal patch features, and nearest-neighbour
scoring. It ships a training pipeline, a TIFF exporter for the official MVTec
evaluation, a FastAPI service, and a Gradio demo.

## Results

Evaluated on the 15 MVTec AD categories (image-level AU-ROC, and AU-PRO for
localisation with a 0.3 FPR integration limit).

| Metric | Mean |
|---|---|
| Image-level AU-ROC | 0.904 |
| AU-PRO (localisation) | 0.917 |

Grid is trained at 256px rather than the default 224px; its fine periodic
texture benefits from the denser feature grid, which raised its AU-ROC from
0.637 to 0.701. Other categories showed no reliable gain from higher resolution
and are trained at 224px. Per-category resolution is configured in
[`src/config.py`](src/config.py).

## Project layout

```
src/
  config.py              Per-category resolution / coreset settings
  dataset.py             MVTec dataset loader and image transforms
  patchcore.py           Backbone, memory bank, coreset, scoring
  train.py               Build a memory bank + calibrate a threshold
  inference.py           Load a model and predict on a single image
  export_anomaly_maps.py Write per-pixel heatmaps as TIFFs for evaluation
  utils.py               Offline analysis helpers
api/app.py               FastAPI inference service
ui/demo.py               Gradio demo
evaluation/              Official MVTec evaluation scripts (vendored)
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
and extract it so each category sits directly under `data/`:

```
data/<category>/train/good/*.png
data/<category>/test/<defect_type>/*.png
data/<category>/ground_truth/<defect_type>/*_mask.png
```

## Usage

### Train

Builds the memory bank for one category and saves it, together with a
`metadata.json` holding the input resolution and a calibrated decision
threshold, under `outputs/memory_bank/<category>/`.

```bash
python -m src.train --data_root data --category grid
```

Resolution and coreset ratio default to the per-category values in
`src/config.py` and can be overridden with `--image_size` / `--coreset_ratio`.
The threshold is set to the 99th percentile of the (defect-free) training
scores, so no magic numbers are needed downstream.

### Export anomaly maps and evaluate

```bash
python -m src.export_anomaly_maps --data_root data --category grid
python evaluation/evaluate_experiment.py \
    --anomaly_maps_dir outputs/anomaly_maps \
    --dataset_base_dir data \
    --output_dir outputs/metrics \
    --evaluated_objects grid
```

`print_metrics.py` renders the resulting `metrics.json` files as tables.

### API

```bash
uvicorn api.app:app --reload
```

- `GET  /categories` — categories with a trained memory bank
- `POST /predict/{category}` — anomaly score and defect flag for an uploaded image
- `POST /predict/{category}/heatmap` — the heatmap overlay as a JPEG

### Demo

```bash
python ui/demo.py
```

Upload an image, pick its category, and the model returns the overlay and score.
Only categories with a trained memory bank appear in the dropdown.

## How it works

- **Features.** Layer2 and layer3 activations of a frozen Wide-ResNet-50-2 are
  concatenated (layer3 upsampled to layer2's resolution). These mid-level maps
  localise defects well; layer1 is too generic and inflates the patch count,
  layer4 is too semantic.
- **Memory bank.** Patch features from all normal images are subsampled with a
  greedy k-center (farthest-point) coreset, which spreads the retained patches
  across the feature manifold far better than uniform sampling. The pool is
  capped and seeded for reproducible builds.
- **Scoring.** Each test patch is scored by its distance to the nearest memory
  bank entry; the maximum patch score is the image-level anomaly score, and the
  per-patch scores form the heatmap.

## Configuration notes

- Memory banks and metadata are written per category under `outputs/`, which is
  git-ignored along with the dataset and generated artifacts.
- Anomaly maps are exported at the original image resolution so they align with
  the full-resolution ground-truth masks used by the evaluation script.

## License

The `evaluation/` directory contains the official MVTec evaluation scripts and
is covered by its own license ([`evaluation/LICENSE.txt`](evaluation/LICENSE.txt)).
