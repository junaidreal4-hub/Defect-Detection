"""Offline analysis helpers, kept out of the inference path."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def find_best_threshold(labels: list, scores: list) -> float:
    """Threshold that maximises Youden's J statistic (TPR - FPR) on labelled data."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def save_result_figure(original_path: str, overlay_bgr: np.ndarray, score: float, save_path: str):
    original = cv2.imread(original_path)
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Anomaly heatmap  |  score: {score:.4f}")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
