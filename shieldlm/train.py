#!/usr/bin/env python3
"""ShieldLM training script for DeBERTa and Llama models.

Usage:
    # DeBERTa-v3-base (2x RTX 3090)
    accelerate launch --num_processes=2 -m shieldlm.train --config configs/deberta_base.yaml

    # DeBERTa-v3-large (2x RTX 3090)
    accelerate launch --num_processes=2 -m shieldlm.train --config configs/deberta_large.yaml

    # Llama QLoRA (single GPU)
    python -m shieldlm.train --config configs/llama_8b_qlora.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_curve
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from shieldlm.utils import compute_all_metrics, save_thresholds

log = logging.getLogger("shieldlm.train")


def compute_metrics(eval_pred):
    """Custom metrics: TPR at fixed FPR operating points.

    This is THE critical metric for model selection — not accuracy, not F1, not AUC.
    See PromptShield (Jacob et al., 2025) for why.
    """
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:, 1]  # P(ATTACK)

    metrics = compute_all_metrics(labels, probs)

    # Also compute accuracy at the 1% FPR threshold
    fpr, tpr, thresholds = roc_curve(labels, probs)
    threshold_1pct = float(np.interp(0.01, fpr, thresholds))
    preds_1pct = (probs >= threshold_1pct).astype(int)
    metrics["accuracy_at_1pct_fpr"] = float((preds_1pct == labels).mean())

    return metrics


def load_and_tokenize(config: dict, tokenizer):
    """Load parquet data and tokenize for HuggingFace Trainer."""
    train_df = pd.read_parquet(config["data"]["train_file"])
    val_df = pd.read_parquet(config["data"]["val_file"])

    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    max_length = config["model"]["max_length"]

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_ds = Dataset.from_pandas(
        train_df[[text_col, label_col]].rename(columns={label_col: "label"})
    )
    val_ds = Dataset.from_pandas(
        val_df[[text_col, label_col]].rename(columns={label_col: "label"})
    )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])

    return train_ds, val_ds


def setup_model_deberta(config: dict):
    """Load DeBERTa model and tokenizer."""
    model_name = config["model"]["name"]
    log.info(f"Loading DeBERTa model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"],
        problem_type=config["model"].get("problem_type", "single_label_classification"),
    )
    return model, tokenizer


def setup_model_llama_qlora(config: dict):
    """Load Llama model with QLoRA (4-bit quantization + LoRA adapters)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    model_name = config["model"]["name"]
    log.info(f"Loading Llama model with QLoRA: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = config["model"]["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_training_args(config: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from YAML config."""
    tc = config["training"]
    return TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=tc["epochs"],
        learning_rate=tc["learning_rate"],
        weight_decay=tc.get("weight_decay", 0.01),
        warmup_ratio=tc.get("warmup_ratio", 0.1),
        lr_scheduler_type=tc.get("lr_scheduler", "cosine"),
        fp16=tc.get("fp16", True),
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 64),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        eval_strategy=tc.get("evaluation_strategy", "steps"),
        eval_steps=tc.get("eval_steps", 200),
        save_strategy=tc.get("save_strategy", "steps"),
        save_steps=tc.get("save_steps", 200),
        load_best_model_at_end=tc.get("load_best_model_at_end", True),
        metric_for_best_model=tc.get("metric_for_best_model", "tpr_at_fpr_01"),
        greater_is_better=tc.get("greater_is_better", True),
        logging_steps=tc.get("logging_steps", 50),
        report_to=tc.get("report_to", "none"),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=tc.get("gradient_checkpointing", False),
    )


def calibrate_thresholds(trainer, val_dataset, output_dir: str) -> dict:
    """Compute calibrated thresholds on validation set and save.

    These thresholds are used by ShieldLMDetector for deployment — users pick
    an FPR target, the detector uses the corresponding decision threshold.
    """
    log.info("Calibrating thresholds on validation set...")
    predictions = trainer.predict(val_dataset)
    probs = softmax(predictions.predictions, axis=1)[:, 1]
    labels = predictions.label_ids

    fpr, tpr, thresholds = roc_curve(labels, probs)
    calibrated = {}
    for fpr_target in [0.001, 0.005, 0.01, 0.05]:
        thresh = float(np.interp(fpr_target, fpr, thresholds))
        tpr_val = float(np.interp(fpr_target, fpr, tpr))
        calibrated[str(fpr_target)] = {
            "threshold": thresh,
            "tpr": tpr_val,
            "fpr_target": fpr_target,
        }
        log.info(f"  FPR={fpr_target}: threshold={thresh:.4f}, TPR={tpr_val:.4f}")

    output_path = Path(output_dir) / "calibrated_thresholds.json"
    save_thresholds(calibrated, str(output_path))
    return calibrated


def main():
    parser = argparse.ArgumentParser(description="Train ShieldLM classifier")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log.info(f"Training with config: {args.config}")
    log.info(f"Model: {config['model']['name']}")

    # Setup model — choose DeBERTa or Llama QLoRA based on config
    is_qlora = config["model"].get("quantization") == "nf4"
    if is_qlora:
        model, tokenizer = setup_model_llama_qlora(config)
    else:
        model, tokenizer = setup_model_deberta(config)

    # Load and tokenize data
    train_ds, val_ds = load_and_tokenize(config, tokenizer)
    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Build training arguments
    training_args = build_training_args(config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    log.info("Starting training...")
    trainer.train()

    # Save best model
    output_dir = config["output"]["dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Model saved to {output_dir}")

    # Calibrate thresholds on validation set
    calibrate_thresholds(trainer, val_ds, output_dir)

    log.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
