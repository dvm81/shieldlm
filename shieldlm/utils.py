"""Shared utilities for ShieldLM training, evaluation, and inference."""

import json
import logging

import numpy as np
from sklearn.metrics import roc_curve

log = logging.getLogger("shieldlm")

# Label constants
LABEL_BENIGN = 0
LABEL_ATTACK = 1
LABEL_NAMES = {0: "BENIGN", 1: "ATTACK"}
CATEGORIES = ["benign", "direct_injection", "indirect_injection", "jailbreak"]

# Default FPR operating points (following PromptShield methodology)
DEFAULT_FPR_TARGETS = [0.001, 0.005, 0.01, 0.05]


def compute_tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, fpr_target: float
) -> tuple[float, float]:
    """Compute TPR and decision threshold at a given FPR operating point.

    Args:
        labels: Ground truth binary labels (0/1).
        scores: Predicted P(ATTACK) scores.
        fpr_target: Target false positive rate.

    Returns:
        (tpr, threshold) at the specified FPR.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    target_tpr = float(np.interp(fpr_target, fpr, tpr))
    target_threshold = float(np.interp(fpr_target, fpr, thresholds))
    return target_tpr, target_threshold


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute all evaluation metrics at standard FPR operating points.

    Returns dict with keys like tpr_at_fpr_001, tpr_at_fpr_01, auc, and
    corresponding thresholds.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(_trapz(tpr, fpr))

    metrics = {"auc": auc}
    for fpr_target in DEFAULT_FPR_TARGETS:
        # Key suffix: 0.001 -> "001", 0.01 -> "01", 0.005 -> "005", 0.05 -> "05"
        key_suffix = str(fpr_target).replace("0.", "").replace(".", "")
        target_tpr = float(np.interp(fpr_target, fpr, tpr))
        target_thresh = float(np.interp(fpr_target, fpr, thresholds))
        metrics[f"tpr_at_fpr_{key_suffix}"] = target_tpr
        metrics[f"threshold_at_fpr_{key_suffix}"] = target_thresh

    return metrics


def save_thresholds(thresholds: dict, path: str):
    """Save calibrated thresholds to JSON."""
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Calibrated thresholds saved to {path}")


def load_thresholds(path: str) -> dict:
    """Load calibrated thresholds from JSON."""
    with open(path) as f:
        return json.load(f)
