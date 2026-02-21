"""Tests for ShieldLMDetector.

All tests use mocked models — no GPU or HuggingFace downloads required.
"""

import sys
from unittest.mock import MagicMock

# Mock torch and transformers before importing detector
# This allows tests to run in CI without GPU dependencies
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
if "transformers" not in sys.modules:
    sys.modules["transformers"] = MagicMock()

import numpy as np
import pytest

from shieldlm.detector import DEFAULT_THRESHOLDS, ShieldLMDetector


@pytest.fixture
def mock_detector():
    """Create a detector with mock model and known thresholds."""
    model = MagicMock()
    tokenizer = MagicMock()
    thresholds = {
        "0.001": {"threshold": 0.9, "tpr": 0.8, "fpr_target": 0.001},
        "0.01": {"threshold": 0.7, "tpr": 0.92, "fpr_target": 0.01},
        "0.05": {"threshold": 0.5, "tpr": 0.97, "fpr_target": 0.05},
    }
    return ShieldLMDetector(model, tokenizer, thresholds, device="cpu")


class TestThresholdLookup:
    def test_exact_match(self, mock_detector):
        assert mock_detector._get_threshold(0.01) == 0.7
        assert mock_detector._get_threshold(0.001) == 0.9
        assert mock_detector._get_threshold(0.05) == 0.5

    def test_closest_fallback(self, mock_detector):
        # 0.005 is not calibrated — should fall back to closest (0.001 or 0.01)
        threshold = mock_detector._get_threshold(0.005)
        assert threshold in [0.9, 0.7]  # Either closest is valid

    def test_default_thresholds_structure(self):
        assert "0.001" in DEFAULT_THRESHOLDS
        assert "0.01" in DEFAULT_THRESHOLDS
        for key, val in DEFAULT_THRESHOLDS.items():
            assert "threshold" in val
            assert "tpr" in val
            assert "fpr_target" in val


class TestDetect:
    def test_output_keys(self, mock_detector):
        """detect() must return label, score, threshold."""
        mock_detector._score = MagicMock(return_value=np.array([0.85]))
        result = mock_detector.detect("test text")
        assert "label" in result
        assert "score" in result
        assert "threshold" in result

    def test_attack_label(self, mock_detector):
        """Score above threshold should return ATTACK."""
        mock_detector._score = MagicMock(return_value=np.array([0.85]))
        result = mock_detector.detect("test", fpr_target=0.01)  # threshold = 0.7
        assert result["label"] == "ATTACK"
        assert result["score"] == 0.85
        assert result["threshold"] == 0.7

    def test_benign_label(self, mock_detector):
        """Score below threshold should return BENIGN."""
        mock_detector._score = MagicMock(return_value=np.array([0.3]))
        result = mock_detector.detect("test", fpr_target=0.01)  # threshold = 0.7
        assert result["label"] == "BENIGN"
        assert result["score"] == 0.3

    def test_boundary_score(self, mock_detector):
        """Score exactly at threshold should return ATTACK (>=)."""
        mock_detector._score = MagicMock(return_value=np.array([0.7]))
        result = mock_detector.detect("test", fpr_target=0.01)
        assert result["label"] == "ATTACK"

    def test_different_fpr_targets(self, mock_detector):
        """Different FPR targets should use different thresholds."""
        mock_detector._score = MagicMock(return_value=np.array([0.85]))

        r1 = mock_detector.detect("test", fpr_target=0.01)   # threshold = 0.7
        r2 = mock_detector.detect("test", fpr_target=0.001)  # threshold = 0.9

        assert r1["label"] == "ATTACK"   # 0.85 >= 0.7
        assert r2["label"] == "BENIGN"   # 0.85 < 0.9


class TestDetectBatch:
    def test_output_length(self, mock_detector):
        """Batch output length should match input length."""
        mock_detector._score = MagicMock(return_value=np.array([0.8, 0.2, 0.9]))
        results = mock_detector.detect_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_each_result_has_keys(self, mock_detector):
        mock_detector._score = MagicMock(return_value=np.array([0.5]))
        results = mock_detector.detect_batch(["test"])
        for r in results:
            assert "label" in r
            assert "score" in r
            assert "threshold" in r

    def test_mixed_labels(self, mock_detector):
        """Batch with mixed scores should produce mixed labels."""
        mock_detector._score = MagicMock(return_value=np.array([0.8, 0.2]))
        results = mock_detector.detect_batch(["attack", "benign"], fpr_target=0.01)
        assert results[0]["label"] == "ATTACK"   # 0.8 >= 0.7
        assert results[1]["label"] == "BENIGN"   # 0.2 < 0.7
