"""Tests for evaluation utilities.

All tests use synthetic data — no GPU or model downloads required.
"""

import numpy as np

from shieldlm.utils import compute_all_metrics, compute_tpr_at_fpr


class TestComputeTprAtFpr:
    def test_perfect_classifier(self):
        """Perfect predictions should yield TPR=1.0 at any FPR."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
        tpr, threshold = compute_tpr_at_fpr(labels, scores, 0.01)
        assert tpr >= 0.9  # Should be near-perfect

    def test_random_classifier(self):
        """Random predictions should yield TPR near FPR."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, size=10000)
        scores = rng.random(size=10000)
        tpr, threshold = compute_tpr_at_fpr(labels, scores, 0.01)
        # Random classifier: TPR ~ FPR
        assert tpr < 0.1

    def test_threshold_is_float(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        _, threshold = compute_tpr_at_fpr(labels, scores, 0.01)
        assert isinstance(threshold, float)


class TestComputeAllMetrics:
    def test_returns_expected_keys(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = compute_all_metrics(labels, scores)

        assert "auc" in metrics
        assert "tpr_at_fpr_001" in metrics
        assert "tpr_at_fpr_01" in metrics
        assert "tpr_at_fpr_05" in metrics
        assert "threshold_at_fpr_001" in metrics
        assert "threshold_at_fpr_01" in metrics

    def test_perfect_classifier_auc(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
        metrics = compute_all_metrics(labels, scores)
        assert metrics["auc"] > 0.95

    def test_random_classifier_auc(self):
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, size=10000)
        scores = rng.random(size=10000)
        metrics = compute_all_metrics(labels, scores)
        assert 0.4 < metrics["auc"] < 0.6

    def test_all_values_are_float(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_all_metrics(labels, scores)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_tpr_monotonic_with_fpr(self):
        """Higher FPR tolerance should yield higher TPR."""
        labels = np.array([0] * 100 + [1] * 100)
        rng = np.random.RandomState(42)
        scores = np.concatenate([rng.random(100) * 0.5, rng.random(100) * 0.5 + 0.5])
        metrics = compute_all_metrics(labels, scores)
        # TPR at 5% FPR should be >= TPR at 1% FPR >= TPR at 0.1% FPR
        assert metrics["tpr_at_fpr_05"] >= metrics["tpr_at_fpr_01"]
        assert metrics["tpr_at_fpr_01"] >= metrics["tpr_at_fpr_001"]
