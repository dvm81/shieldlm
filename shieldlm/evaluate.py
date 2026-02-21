#!/usr/bin/env python3
"""ShieldLM evaluation framework with FPR-based metrics.

Reports TPR at fixed FPR operating points (not AUC) following PromptShield
methodology (Jacob et al., 2025). Includes per-category breakdown, FPR by
benign data type, latency profiling, and baseline comparisons.

Usage:
    python -m shieldlm.evaluate \
        --model models/deberta-v3-base-shieldlm \
        --test-data data/unified/test.parquet \
        --output results/eval_report.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import roc_curve
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shieldlm.utils import compute_all_metrics

log = logging.getLogger("shieldlm.evaluate")

# Baseline models for comparison
BASELINES = {
    "protectai_v2": {
        "model": "protectai/deberta-v3-base-prompt-injection-v2",
        "type": "huggingface",
    },
    "promptguard_86m": {
        "model": "meta-llama/Prompt-Guard-86M",
        "type": "huggingface",
    },
}

# Source-to-data-type mapping for FPR breakdown
# (conversational benign vs application-structured benign vs sensitive-topic benign)
BENIGN_DATA_TYPE_MAP = {
    "conversational": [
        "alespalla/chatbot-instruction-prompts",
        "jackhhao/jailbreak-classification",
    ],
    "application_structured": [
        "injecagent/clean_response",  # Prefix match
    ],
    "sensitive_topic": [
        "jailbreakbench/jbb-behaviors",
    ],
}


class ShieldLMEvaluator:
    """Deployment-aware evaluation following PromptShield methodology."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.model_path = model_path

    def _predict_scores(
        self, texts: list[str], batch_size: int = 32, max_length: int = 512
    ) -> np.ndarray:
        """Get P(ATTACK) scores for a list of texts."""
        all_scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()
            probs = softmax(logits, axis=1)[:, 1]
            all_scores.extend(probs)
        return np.array(all_scores)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Full evaluation with per-category and per-data-type breakdowns.

        Returns:
            {
                "model": str,
                "overall": {tpr_at_fpr_001, tpr_at_fpr_01, auc, n_samples, ...},
                "by_category": {category: {tpr, n}, ...},
                "fpr_by_data_type": {type: fpr, ...},
                "latency": {mean_ms, p95_ms, p99_ms},
            }
        """
        texts = test_df["text"].tolist()
        labels = test_df["label_binary"].values

        # Get scores
        scores = self._predict_scores(texts)

        # Overall metrics
        overall = compute_all_metrics(labels, scores)
        overall["n_samples"] = len(labels)
        overall["n_attack"] = int((labels == 1).sum())
        overall["n_benign"] = int((labels == 0).sum())

        # Compute threshold at 1% FPR for per-category and per-type breakdowns
        fpr, tpr, thresholds = roc_curve(labels, scores)
        threshold_1pct = float(np.interp(0.01, fpr, thresholds))

        # Per-category breakdown
        by_category = self._compute_per_category(test_df, scores, threshold_1pct)

        # FPR by data type
        fpr_by_type = self._compute_fpr_by_data_type(test_df, scores, threshold_1pct)

        # Latency
        latency = self._measure_latency(texts[:100])

        return {
            "model": self.model_path,
            "overall": overall,
            "by_category": by_category,
            "fpr_by_data_type": fpr_by_type,
            "latency": latency,
        }

    def _compute_per_category(
        self, df: pd.DataFrame, scores: np.ndarray, threshold: float
    ) -> dict:
        """Compute TPR per attack category at a given threshold."""
        result = {}
        for cat in ["direct_injection", "indirect_injection", "jailbreak"]:
            mask = df["label_category"].values == cat
            if mask.sum() == 0:
                continue
            cat_scores = scores[mask]
            cat_preds = (cat_scores >= threshold).astype(int)
            result[cat] = {
                "tpr": float(cat_preds.mean()),
                "n": int(mask.sum()),
            }
        return result

    def _compute_fpr_by_data_type(
        self, df: pd.DataFrame, scores: np.ndarray, threshold: float
    ) -> dict:
        """Compute FPR separately for different benign data types.

        This is essential for understanding deployment behavior — a classifier
        might have low overall FPR but high FPR on application-structured data
        (JSON/tool responses), making it unusable for agent pipelines.
        """
        benign_mask = df["label_binary"].values == 0
        result = {}

        for data_type, source_patterns in BENIGN_DATA_TYPE_MAP.items():
            # Match sources by exact match or prefix
            type_mask = np.zeros(len(df), dtype=bool)
            for pattern in source_patterns:
                if pattern.endswith("/clean_response"):
                    # Prefix match for InjecAgent clean responses
                    type_mask |= df["source"].str.startswith(pattern, na=False).values
                else:
                    type_mask |= (df["source"].values == pattern)

            combined_mask = benign_mask & type_mask
            if combined_mask.sum() == 0:
                continue

            preds = (scores[combined_mask] >= threshold).astype(int)
            result[data_type] = {
                "fpr": float(preds.mean()),
                "n": int(combined_mask.sum()),
            }

        return result

    def _measure_latency(self, texts: list[str]) -> dict:
        """Measure single-sample inference latency."""
        # Warmup
        if texts:
            inputs = self.tokenizer(
                texts[0], truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)

        times = []
        for text in texts:
            start = time.perf_counter()
            inputs = self.tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        if not times:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        return {
            "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }

    def compare_baselines(self, test_df: pd.DataFrame) -> dict:
        """Benchmark against baseline models."""
        results = {}
        for name, config in BASELINES.items():
            log.info(f"Evaluating baseline: {name} ({config['model']})")
            try:
                evaluator = ShieldLMEvaluator(config["model"], device=self.device)
                results[name] = evaluator.evaluate(test_df)
            except Exception as e:
                log.warning(f"Failed to evaluate {name}: {e}")
                results[name] = {"error": str(e)}
        return results


def format_comparison_table(shieldlm_results: dict, baseline_results: dict) -> str:
    """Format results as a markdown comparison table."""
    rows = []

    # ShieldLM
    o = shieldlm_results["overall"]
    lat = shieldlm_results["latency"]
    rows.append(
        f"| ShieldLM (DeBERTa-base) | 86M | "
        f"{o.get('tpr_at_fpr_001', 0) * 100:.1f}% | "
        f"{o.get('tpr_at_fpr_01', 0) * 100:.1f}% | "
        f"{lat['mean_ms']:.1f}ms |"
    )

    # Baselines
    for name, result in baseline_results.items():
        if "error" in result:
            rows.append(f"| {name} | — | error | error | — |")
            continue
        o = result["overall"]
        lat = result["latency"]
        rows.append(
            f"| {name} | 86M | "
            f"{o.get('tpr_at_fpr_001', 0) * 100:.1f}% | "
            f"{o.get('tpr_at_fpr_01', 0) * 100:.1f}% | "
            f"{lat['mean_ms']:.1f}ms |"
        )

    header = "| Model | Params | TPR@0.1%FPR | TPR@1%FPR | Latency |\n"
    header += "|-------|--------|-------------|-----------|---------|"
    return header + "\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShieldLM classifier")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test parquet")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparisons")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test_data)
    log.info(f"Test set: {len(test_df)} samples")

    evaluator = ShieldLMEvaluator(args.model, device=args.device)
    results = evaluator.evaluate(test_df)

    # Print summary
    o = results["overall"]
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  TPR @ 0.1% FPR: {o.get('tpr_at_fpr_001', 0) * 100:.1f}%")
    log.info(f"  TPR @ 1.0% FPR: {o.get('tpr_at_fpr_01', 0) * 100:.1f}%")
    log.info(f"  AUC:            {o.get('auc', 0):.4f}")
    log.info(f"  Samples:        {o['n_samples']} ({o['n_attack']} attack, {o['n_benign']} benign)")

    log.info("\nPer-category TPR (at 1% FPR threshold):")
    for cat, vals in results["by_category"].items():
        log.info(f"  {cat}: {vals['tpr'] * 100:.1f}% ({vals['n']} samples)")

    log.info("\nFPR by benign data type (at 1% FPR threshold):")
    for dtype, vals in results["fpr_by_data_type"].items():
        log.info(f"  {dtype}: {vals['fpr'] * 100:.2f}% ({vals['n']} samples)")

    log.info(f"\nLatency (single sample): {results['latency']['mean_ms']:.1f}ms "
             f"(p95: {results['latency']['p95_ms']:.1f}ms)")

    # Baseline comparisons
    if args.baselines:
        log.info("\nRunning baseline comparisons...")
        baseline_results = evaluator.compare_baselines(test_df)
        results["baselines"] = baseline_results
        table = format_comparison_table(results, baseline_results)
        log.info(f"\n{table}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
