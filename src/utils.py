import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import cv2


def compute_auroc(labels: list, scores: list) -> float:
    """AUROC is the standard metric for anomaly detection benchmarks."""
    return roc_auc_score(labels, scores)


def find_best_threshold(labels: list, scores: list) -> float:
    """
    Finds the threshold that maximizes Youden's J statistic (TPR - FPR).
    Better than just picking 0.5 arbitrarily.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])

def compute_threshold_from_training(model, train_loader, device="cpu", percentile=99):
    """
    Runs inference on the training set (all normal images) and sets the
    threshold at a high percentile of their scores. Anything above what
    normal images typically score gets flagged as anomalous.
    """
    scores = []
    for images, _, _ in train_loader:
        for img in images:
            score, _ = model.score(img.unsqueeze(0))
            scores.append(score)

    threshold = float(np.percentile(scores, percentile))
    print(f"Training score stats — min: {min(scores):.3f}, max: {max(scores):.3f}, mean: {np.mean(scores):.3f}")
    print(f"Threshold set at {percentile}th percentile: {threshold:.3f}")
    return threshold

def save_result_figure(original_path: str, overlay_bgr: np.ndarray, score: float, save_path: str):
    original = cv2.imread(original_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Anomaly Heatmap  |  Score: {score:.4f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()